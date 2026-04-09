#include "voxel.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

AppState gApp;

// ===================================================================
//  Internal constants
// ===================================================================
static constexpr UINT  FRAME_COUNT = 2;
static constexpr UINT  CB_ALIGN = 256;
static constexpr int   WATER_LEVEL = 50;

// Staging: 256-byte aligned row pitch for chunk width
static constexpr UINT  STAGING_ROW = (CHUNK_X + 255u) & ~255u;  // 256
static constexpr UINT  STAGING_SLICE = STAGING_ROW * CHUNK_Y;      // 32768
static constexpr UINT  CHUNK_STAGING = STAGING_SLICE * CHUNK_Z;    // 1 048 576

static constexpr int   MAX_UPLOADS_PER_FRAME = 8;
static constexpr UINT  FRAME_STAGING_SIZE = MAX_UPLOADS_PER_FRAME * CHUNK_STAGING;
static constexpr int   GEN_WORKER_COUNT = 2;   // number of terrain-gen worker threads (adjustable)
static constexpr int   MAX_CLEARS_PER_FRAME = 32;  // atlas voxel-data clears recorded per frame

// ---- Occupancy-map staging (one 1×SECTIONS_PER_CHUNK×1 column per chunk) ----
static constexpr UINT  OCC_ROW = 256;                                        // D3D12 row-pitch alignment
static constexpr UINT  CHUNK_OCC_STAGING = OCC_ROW * SECTIONS_PER_CHUNK;     // 2048 – also 512-aligned for placement
static constexpr UINT  FRAME_OCC_STAGING_SIZE = MAX_UPLOADS_PER_FRAME * CHUNK_OCC_STAGING;

static constexpr int REGION_CHUNK_SHIFT = 3;     // log2(REGION_CHUNKS)

static int ChunkToRegion(int c) { return c >> REGION_CHUNK_SHIFT; }

// ===================================================================
//  Frame constants (HLSL mirror)
// ===================================================================
struct FrameConstants {
    float    camPos[3];  float _p0;
    float    fwd[3];     float _p1;
    float    right[3];   float _p2;
    float    up[3];      float _p3;
    uint32_t screenW, screenH;
    int32_t  loadMinCX, loadMinCZ;
    int32_t  loadMaxCX, loadMaxCZ;
    float    tanHalfFov, maxDist;
};
static_assert(sizeof(FrameConstants) <= CB_ALIGN);

// ===================================================================
//  Chunk data structures (file-local)
// ===================================================================
namespace {

    struct ChunkSection {
        uint8_t  blocks[CHUNK_X * SECTION_Y * CHUNK_Z]{};
        uint16_t solidCount = 0;
    };

    struct Chunk {
        std::unique_ptr<ChunkSection> sections[SECTIONS_PER_CHUNK];
        uint8_t surfaceY[CHUNK_X * CHUNK_Z]{};

        uint8_t getBlock(int lx, int ly, int lz) const {
            if (ly < 0 || ly >= CHUNK_Y) return BLOCK_AIR;
            int si = ly >> 4;
            if (!sections[si]) return BLOCK_AIR;
            return sections[si]->blocks[lx + (ly & 15) * CHUNK_X
                + lz * SECTION_Y * CHUNK_X];
        }

        void setBlock(int lx, int ly, int lz, uint8_t bt) {
            if (ly < 0 || ly >= CHUNK_Y) return;
            int si = ly >> 4;
            int idx = lx + (ly & 15) * CHUNK_X + lz * SECTION_Y * CHUNK_X;

            if (!sections[si]) {
                if (bt == BLOCK_AIR) return;
                sections[si] = std::make_unique<ChunkSection>();
            }
            auto& s = *sections[si];
            uint8_t old = s.blocks[idx];
            if (old == bt) return;
            if (old == BLOCK_AIR) ++s.solidCount;
            if (bt == BLOCK_AIR) --s.solidCount;
            s.blocks[idx] = bt;
            if (s.solidCount == 0) sections[si].reset();
        }
    };

    // ---- chunk key helpers ----
    inline int64_t ChunkKey(int cx, int cz) {
        return (int64_t(cx) << 32) | (int64_t(cz) & 0xFFFFFFFF);
    }
    inline int AtlasSlot(int c) {
        return ((c % LOAD_CHUNKS) + LOAD_CHUNKS) % LOAD_CHUNKS;
    }

    // ---- chunk manager state ----
    struct ChunkMgr {
        std::unordered_map<int64_t, Chunk> chunks;
        int centerRX = INT_MIN, centerRZ = INT_MIN;
        int64_t slotKeys[LOAD_CHUNKS][LOAD_CHUNKS];

        std::deque<std::pair<int, int>> uploadQueue;

        // Split clear path:
        //   occClearQueue   – tiny 1×8×1 occupancy columns, drained fully every frame
        //   atlasClearQueue – 32×128×32 voxel blocks, amortised over several frames
        std::deque<std::pair<int, int>> occClearQueue;
        std::deque<std::pair<int, int>> atlasClearQueue;

        void init() {
            centerRX = centerRZ = INT_MIN;
            for (auto& row : slotKeys)
                for (auto& k : row) k = INT64_MIN;
        }
    } cm;

    // =================================================================
    //  Background terrain-generation worker pool
    // =================================================================
    struct GenJob {
        int      cx, cz;
        uint32_t epoch;                  // pool epoch at enqueue time
    };

    struct GenResult {
        int                    cx, cz;
        uint32_t               epoch;
        std::unique_ptr<Chunk> chunk;    // fully built, owned by the result
    };

    struct GenWorkerPool {
        std::vector<std::thread>    threads;

        std::mutex                  mtx;     // guards `jobs` and `results`
        std::condition_variable     cv;
        std::deque<GenJob>          jobs;
        std::deque<GenResult>       results;

        std::atomic<uint32_t>       epoch{ 0 };
        std::atomic<bool>           stop{ false };

        // MAIN-THREAD ONLY: chunk keys currently queued or in flight for the
        // *current* epoch.  Cleared wholesale on every region transition.
        std::unordered_set<int64_t> pending;
    } gen;

} // anon namespace

// ===================================================================
//  GPU state (file-local)
// ===================================================================
static struct GpuState {
    ComPtr<ID3D12Device>              device;
    ComPtr<ID3D12CommandQueue>        queue;
    ComPtr<IDXGISwapChain3>           swapChain;
    ComPtr<ID3D12CommandAllocator>    cmdAlloc[FRAME_COUNT];
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Fence>               fence;
    HANDLE                            fenceEvent = nullptr;
    UINT64                            fenceValues[FRAME_COUNT]{};
    UINT                              frameIndex = 0;

    ComPtr<ID3D12DescriptorHeap> srvHeap;
    UINT                         srvDescSize = 0;

    ComPtr<ID3D12RootSignature>  rootSig;
    ComPtr<ID3D12PipelineState>  pso;

    ComPtr<ID3D12Resource> atlas;                   // Texture3D 768×128×768
    ComPtr<ID3D12Resource> occupancy;               // Texture3D  24×  8× 24  (section occupancy)
    ComPtr<ID3D12Resource> outputTex;
    ComPtr<ID3D12Resource> backBuffers[FRAME_COUNT];
    ComPtr<ID3D12Resource> cbUpload;
    ComPtr<ID3D12Resource> zeroStaging;             // 1 chunk, permanently zeroed
    ComPtr<ID3D12Resource> zeroOccStaging;          // 1 occupancy column, permanently zeroed
    uint8_t* cbMapped = nullptr;

    ComPtr<ID3D12Resource> frameStaging[FRAME_COUNT];
    uint8_t* frameStagingMapped[FRAME_COUNT]{};

    ComPtr<ID3D12Resource> frameOccStaging[FRAME_COUNT];
    uint8_t* frameOccStagingMapped[FRAME_COUNT]{};
} gpu;

// ===================================================================
//  Error helper
// ===================================================================
static void Check(HRESULT hr, const char* m) {
    if (FAILED(hr)) { OutputDebugStringA(m); throw std::runtime_error(m); }
}

// ===================================================================
//  Noise (unchanged CPU-side terrain authoring)
// ===================================================================
static float Lerp(float a, float b, float t) { return a + (b - a) * t; }
static float Smooth(float t) { return t * t * (3.0f - 2.0f * t); }

static uint32_t Hash2D(int x, int y) {
    uint32_t h = 2166136261u;
    h = (h ^ uint32_t(x)) * 16777619u;
    h = (h ^ uint32_t(y)) * 16777619u;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    return h;
}
static float Random01(int x, int y) {
    return float(Hash2D(x, y) & 0x00FFFFFF) / float(0x00FFFFFF);
}

static float ValueNoise(float x, float y) {
    int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    float sx = Smooth(x - float(x0)), sy = Smooth(y - float(y0));
    float a = Lerp(Random01(x0, y0), Random01(x0 + 1, y0), sx);
    float b = Lerp(Random01(x0, y0 + 1), Random01(x0 + 1, y0 + 1), sx);
    return Lerp(a, b, sy);
}

static float FBM(float x, float y, int oct) {
    float s = 0, a = 0.5f, f = 1, n = 0;
    for (int i = 0; i < oct; ++i) {
        s += ValueNoise(x * f, y * f) * a;
        n += a; a *= 0.5f; f *= 2.0f;
    }
    return s / n;
}

// ===================================================================
//  Per-chunk terrain generation – PURE.  Builds and returns a Chunk
//  without touching any shared state so it is safe to call from any
//  worker thread concurrently.
// ===================================================================
static std::unique_ptr<Chunk> BuildChunkTerrain(int cx, int cz) {
    auto chunk = std::make_unique<Chunk>();

    int wx0 = cx * CHUNK_X;
    int wz0 = cz * CHUNK_Z;

    for (int lz = 0; lz < CHUNK_Z; ++lz) {
        int wz = wz0 + lz;
        for (int lx = 0; lx < CHUNK_X; ++lx) {
            int wx = wx0 + lx;

            float large = FBM(wx * 0.0012f, wz * 0.0012f, 5);
            float medium = FBM(wx * 0.0035f, wz * 0.0035f, 6);
            float detail = FBM(wx * 0.012f, wz * 0.012f, 4);

            float e = 0.55f * medium + 0.30f * large + 0.15f * detail;
            e = std::pow(e, 1.35f);
            float ridge = std::fabs(FBM(wx * 0.006f, wz * 0.006f, 5) - 0.5f) * 2.0f;
            int surfH = std::clamp(int(20.0f + e * 170.0f + ridge * 18.0f), 0, CHUNK_Y - 1);

            chunk->surfaceY[lz * CHUNK_X + lx] = uint8_t(surfH);

            BlockType surfType;
            if (surfH < 58)  surfType = BLOCK_SAND;
            else if (surfH < 140) surfType = BLOCK_GRASS;
            else if (surfH < 180) surfType = BLOCK_ROCK;
            else                  surfType = BLOCK_SNOW;

            for (int y = 0; y <= surfH; ++y) {
                BlockType bt;
                if (y < surfH - 4) bt = BLOCK_STONE;
                else if (y < surfH)     bt = BLOCK_DIRT;
                else                    bt = surfType;
                chunk->setBlock(lx, y, lz, bt);
            }
            for (int y = surfH + 1; y <= WATER_LEVEL && y < CHUNK_Y; ++y)
                chunk->setBlock(lx, y, lz, y < 42 ? BLOCK_DEEP_WATER : BLOCK_WATER);
        }
    }
    return chunk;
}

// ===================================================================
//  Worker loop – runs on every generation thread.
// ===================================================================
static void GenWorkerLoop() {
    for (;;) {
        GenJob job;
        {
            std::unique_lock<std::mutex> lk(gen.mtx);
            gen.cv.wait(lk, [] { return gen.stop.load() || !gen.jobs.empty(); });
            if (gen.stop.load()) return;
            job = gen.jobs.front();
            gen.jobs.pop_front();
        }

        // Drop stale jobs before doing expensive noise work.
        if (job.epoch != gen.epoch.load(std::memory_order_relaxed))
            continue;

        std::unique_ptr<Chunk> c = BuildChunkTerrain(job.cx, job.cz);

        // Drop the finished chunk if it became stale during generation.
        if (job.epoch != gen.epoch.load(std::memory_order_relaxed))
            continue;

        std::lock_guard<std::mutex> lk(gen.mtx);
        gen.results.push_back({ job.cx, job.cz, job.epoch, std::move(c) });
    }
}

static void StartGenWorkers(int n = GEN_WORKER_COUNT) {
    if (!gen.threads.empty()) return;          // idempotent
    gen.stop = false;
    gen.threads.reserve(size_t(n));
    for (int i = 0; i < n; ++i)
        gen.threads.emplace_back(GenWorkerLoop);
}

static void StopGenWorkers() {
    if (gen.threads.empty()) return;
    gen.stop = true;
    gen.cv.notify_all();
    for (auto& t : gen.threads) t.join();
    gen.threads.clear();

    std::lock_guard<std::mutex> lk(gen.mtx);
    gen.jobs.clear();
    gen.results.clear();
    gen.pending.clear();
}

// ===================================================================
//  Surface height query (for camera ground-follow)
// ===================================================================
static uint8_t SurfaceHeightAt(float wx, float wz) {
    int bx = int(std::floor(wx)), bz = int(std::floor(wz));
    int cx = bx >> 5, cz = bz >> 5;
    int lx = bx & 31, lz = bz & 31;
    auto it = cm.chunks.find(ChunkKey(cx, cz));
    if (it == cm.chunks.end()) return 0;
    return it->second.surfaceY[lz * CHUNK_X + lx];
}

// ===================================================================
//  Chunk streaming – determine which chunks to load/unload
// ===================================================================
static constexpr int REGION_BLOCK_SHIFT = 8;   // log2(REGION_CHUNKS * CHUNK_X) = log2(256)

static int BlockToRegion(int b) { return b >> REGION_BLOCK_SHIFT; }

static bool RegionInBounds(int rx, int rz, int cRX, int cRZ) {
    return rx >= cRX - 1 && rx <= cRX + 1 &&
        rz >= cRZ - 1 && rz <= cRZ + 1;
}

static void EvictRegion(int rx, int rz) {
    int cxMin = rx * REGION_CHUNKS, czMin = rz * REGION_CHUNKS;
    for (int cx = cxMin; cx < cxMin + REGION_CHUNKS; ++cx)
        for (int cz = czMin; cz < czMin + REGION_CHUNKS; ++cz) {
            int64_t key = ChunkKey(cx, cz);
            int sx = AtlasSlot(cx), sz = AtlasSlot(cz);
            if (cm.slotKeys[sx][sz] == key) {
                cm.slotKeys[sx][sz] = INT64_MIN;
                cm.occClearQueue.push_back({ cx, cz });     // cleared this frame
                cm.atlasClearQueue.push_back({ cx, cz });   // amortised
            }
            cm.chunks.erase(key);
        }
}

static void ScheduleRegionTransition(int newRX, int newRZ) {
    int oldRX = cm.centerRX, oldRZ = cm.centerRZ;

    // ---- Invalidate ALL outstanding generation work immediately ----
    {
        std::lock_guard<std::mutex> lk(gen.mtx);
        gen.epoch.fetch_add(1, std::memory_order_relaxed);
        gen.jobs.clear();
        gen.results.clear();
    }
    gen.pending.clear();

    // ---- Evict regions that left the 3×3 window ----
    if (oldRX != INT_MIN) {
        for (int rx = oldRX - 1; rx <= oldRX + 1; ++rx)
            for (int rz = oldRZ - 1; rz <= oldRZ + 1; ++rz)
                if (!RegionInBounds(rx, rz, newRX, newRZ))
                    EvictRegion(rx, rz);
    }

    cm.centerRX = newRX;
    cm.centerRZ = newRZ;

    // ---- Collect generation jobs for the new window ----
    std::vector<GenJob> newJobs;
    const uint32_t e = gen.epoch.load(std::memory_order_relaxed);

    for (int rx = newRX - 1; rx <= newRX + 1; ++rx)
        for (int rz = newRZ - 1; rz <= newRZ + 1; ++rz) {
            int cxMin = rx * REGION_CHUNKS;
            int czMin = rz * REGION_CHUNKS;
            for (int cx = cxMin; cx < cxMin + REGION_CHUNKS; ++cx)
                for (int cz = czMin; cz < czMin + REGION_CHUNKS; ++cz) {
                    int64_t key = ChunkKey(cx, cz);
                    if (cm.chunks.count(key)) {
                        int sx = AtlasSlot(cx), sz = AtlasSlot(cz);
                        if (cm.slotKeys[sx][sz] != key) {
                            cm.slotKeys[sx][sz] = key;
                            cm.uploadQueue.push_back({ cx, cz });
                        }
                    }
                    else {
                        gen.pending.insert(key);
                        newJobs.push_back({ cx, cz, e });
                    }
                }
        }

    // ---- Sort: nearest chunks to the camera first ----
    int pcx = int(std::floor(gApp.camX)) >> 5;
    int pcz = int(std::floor(gApp.camY)) >> 5;
    std::sort(newJobs.begin(), newJobs.end(),
        [pcx, pcz](const GenJob& a, const GenJob& b) {
            int dxa = a.cx - pcx, dza = a.cz - pcz;
            int dxb = b.cx - pcx, dzb = b.cz - pcz;
            return (dxa * dxa + dza * dza) < (dxb * dxb + dzb * dzb);
        });

    // ---- Hand the whole batch to the worker pool in one locked push ----
    {
        std::lock_guard<std::mutex> lk(gen.mtx);
        for (auto& j : newJobs)
            gen.jobs.push_back(j);
    }
    gen.cv.notify_all();
}

// ===================================================================
//  Drain the worker completion queue on the MAIN thread.  This is the
//  only place that inserts into cm.chunks / slotKeys / uploadQueue, so
//  all GPU-facing state mutation remains single-threaded.
// ===================================================================
static void ProcessGenResults() {
    std::deque<GenResult> local;
    {
        std::lock_guard<std::mutex> lk(gen.mtx);
        local.swap(gen.results);
    }

    const uint32_t curEpoch = gen.epoch.load(std::memory_order_relaxed);

    for (auto& r : local) {
        if (r.epoch != curEpoch)                 // stale – pending already wiped on transition
            continue;

        int64_t key = ChunkKey(r.cx, r.cz);
        gen.pending.erase(key);

        int crx = ChunkToRegion(r.cx), crz = ChunkToRegion(r.cz);
        if (!RegionInBounds(crx, crz, cm.centerRX, cm.centerRZ))
            continue;                            // defensive – should not happen for curEpoch
        if (cm.chunks.count(key))
            continue;                            // already present (duplicate guard)

        cm.chunks.emplace(key, std::move(*r.chunk));

        int sx = AtlasSlot(r.cx), sz = AtlasSlot(r.cz);
        cm.slotKeys[sx][sz] = key;
        cm.uploadQueue.push_back({ r.cx, r.cz });
    }
}

// ===================================================================
//  GenerateTerrain – called from Demo.cpp before InitD3D12
// ===================================================================
void GenerateTerrain() {
    cm.init();
    StartGenWorkers(GEN_WORKER_COUNT);

    int prx = BlockToRegion(int(std::floor(gApp.camX)));
    int prz = BlockToRegion(int(std::floor(gApp.camY)));
    cm.centerRX = prx;
    cm.centerRZ = prz;

    // Enqueue all 576 chunks to the worker pool.
    std::vector<GenJob> jobs;
    const uint32_t e = gen.epoch.load(std::memory_order_relaxed);
    for (int rx = prx - 1; rx <= prx + 1; ++rx)
        for (int rz = prz - 1; rz <= prz + 1; ++rz) {
            int cxMin = rx * REGION_CHUNKS, czMin = rz * REGION_CHUNKS;
            for (int cx = cxMin; cx < cxMin + REGION_CHUNKS; ++cx)
                for (int cz = czMin; cz < czMin + REGION_CHUNKS; ++cz) {
                    gen.pending.insert(ChunkKey(cx, cz));
                    jobs.push_back({ cx, cz, e });
                }
        }
    {
        std::lock_guard<std::mutex> lk(gen.mtx);
        for (auto& j : jobs) gen.jobs.push_back(j);
    }
    gen.cv.notify_all();

    // Block until every chunk has been generated AND handed back.
    // Main thread does no noise work – it only drains the completion queue.
    while (!gen.pending.empty()) {
        ProcessGenResults();
        if (!gen.pending.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    float sh = float(SurfaceHeightAt(gApp.camX, gApp.camY));
    gApp.camZ = sh + 2.5f;
}

// ===================================================================
//  Flatten sparse chunk into staging-buffer layout
// ===================================================================
static void FlattenChunk(const Chunk& c, uint8_t* dst) {
    std::memset(dst, BLOCK_AIR, CHUNK_STAGING);
    for (int si = 0; si < SECTIONS_PER_CHUNK; ++si) {
        if (!c.sections[si]) continue;
        const auto& sec = *c.sections[si];
        for (int lz = 0; lz < CHUNK_Z; ++lz)
            for (int lyl = 0; lyl < SECTION_Y; ++lyl) {
                int wy = si * SECTION_Y + lyl;
                std::memcpy(
                    dst + size_t(lz) * STAGING_SLICE
                    + size_t(wy) * STAGING_ROW,
                    &sec.blocks[lyl * CHUNK_X + lz * SECTION_Y * CHUNK_X],
                    CHUNK_X);
            }
    }
}

// ===================================================================
//  Flatten chunk section-occupancy into staging-buffer layout.
//  Footprint is 1 × SECTIONS_PER_CHUNK × 1 with RowPitch = OCC_ROW, so
//  section si lives at byte offset si * OCC_ROW.
// ===================================================================
static void FlattenOccupancy(const Chunk& c, uint8_t* dst) {
    std::memset(dst, 0, CHUNK_OCC_STAGING);
    for (int si = 0; si < SECTIONS_PER_CHUNK; ++si)
        dst[size_t(si) * OCC_ROW] = c.sections[si] ? 1u : 0u;
}

// ===================================================================
//  HLSL compute shader – two-level (section / voxel) DDA through
//  toroidal atlas, using a section-occupancy map to skip empty
//  32×16×32 regions in a single step.
// ===================================================================
static const char* g_shaderSrc = R"(
cbuffer CB : register(b0)
{
    float3 camPos;   float _p0;
    float3 camFwd;   float _p1;
    float3 camRight; float _p2;
    float3 camUp;    float _p3;
    uint   screenW, screenH;
    int    loadMinCX, loadMinCZ;
    int    loadMaxCX, loadMaxCZ;
    float  tanHalfFov, maxDist;
};

Texture3D<uint>     Atlas     : register(t0);
Texture3D<uint>     Occupancy : register(t1);
RWTexture2D<float4> Output    : register(u0);

static const int   LC   = 24;
static const int   CX   = 32;
static const int   CY   = 128;
static const int   SY   = 16;
static const int   SPC  = 8;

static const int3   SEC_I = int3(CX, SY, CX);
static const float3 SEC_F = float3(32.0, 16.0, 32.0);

static const int   MAX_COARSE = 64;
static const int   MAX_FINE   = 96;

static const float3 FOG_CLR = float3(0.67, 0.80, 0.92);

static const float3 bcolors[9] = {
    float3(0,0,0),
    float3(0.50,0.50,0.50),
    float3(0.55,0.36,0.18),
    float3(0.30,0.58,0.16),
    float3(0.76,0.70,0.50),
    float3(0.15,0.40,0.68),
    float3(0.91,0.91,0.96),
    float3(0.05,0.14,0.38),
    float3(0.42,0.42,0.39),
};
static const float faceBri[6] = { 0.80, 0.80, 1.00, 0.45, 0.65, 0.65 };

// Outward surface normal per hit-face index.
static const float3 faceN[6] = {
    float3( 1, 0, 0), float3(-1, 0, 0),
    float3( 0, 1, 0), float3( 0,-1, 0),
    float3( 0, 0, 1), float3( 0, 0,-1)
};
// Deterministic axis-aligned tangent / bitangent per face.
static const float3 faceT[6] = {
    float3( 0, 0, 1), float3( 0, 0, 1),
    float3( 1, 0, 0), float3( 1, 0, 0),
    float3( 1, 0, 0), float3( 1, 0, 0)
};
static const float3 faceB[6] = {
    float3( 0, 1, 0), float3( 0, 1, 0),
    float3( 0, 0, 1), float3( 0, 0, 1),
    float3( 0, 1, 0), float3( 0, 1, 0)
};

// ---- Ambient-occlusion parameters ----
static const int   AO_SAMPLES  = 6;
static const float AO_MAX_DIST = 12.0;
static const float AO_BIAS     = 0.02;
static const float AO_STRENGTH = 0.9;

// Directional influence: sun-facing surfaces receive less AO shadowing.
// Set AO_SUN_INFLUENCE to 0.0 for omnidirectional AO (original behaviour).
static const float3 AO_SUN_DIR       = float3(0.8305, 0.4983, 0.2491); // morning sun high in east
static const float  AO_SUN_INFLUENCE = 1.0;   // 0 = omni, 1 = full directional
static const float  AO_MIN_BRIGHT    = 0.12;  // floor to prevent pure-black backfaces

// 6 fixed hemisphere directions in tangent space (z = along normal).
static const float3 AO_DIRS[6] = {
    float3( 0.87543,  0.23457, 0.42262),
    float3(-0.64085,  0.64085, 0.42262),
    float3(-0.23457, -0.87543, 0.42262),
    float3( 0.12941,  0.48296, 0.86603),
    float3(-0.48296, -0.12941, 0.86603),
    float3( 0.35355, -0.35355, 0.86603)
};

uint PosMod(int v, int m) { return uint(((v % m) + m) % m); }

int3 SectionOf(int3 p) { return int3(p.x >> 5, p.y >> 4, p.z >> 5); }

float3 InitTMax(int3 cell, float3 cellSize, float3 origin, int3 st, float3 tD)
{
    float3 lo = float3(cell) * cellSize;
    float3 hi = lo + cellSize;
    float3 d;
    d.x = (st.x > 0) ? (hi.x - origin.x) : (origin.x - lo.x);
    d.y = (st.y > 0) ? (hi.y - origin.y) : (origin.y - lo.y);
    d.z = (st.z > 0) ? (hi.z - origin.z) : (origin.z - lo.z);
    return d * tD;
}

uint GetBlock(int3 wp)
{
    if (wp.y < 0 || wp.y >= CY) return 0u;
    int cx = wp.x >> 5;
    int cz = wp.z >> 5;
    if (cx < loadMinCX || cx >= loadMaxCX ||
        cz < loadMinCZ || cz >= loadMaxCZ) return 0u;
    uint sx = PosMod(cx, LC);
    uint sz = PosMod(cz, LC);
    uint lx = uint(wp.x) & 31u;
    uint lz = uint(wp.z) & 31u;
    return Atlas.Load(int4(sx * CX + lx, wp.y, sz * CX + lz, 0));
}

uint GetOccupancy(int3 sc)
{
    if (sc.y < 0 || sc.y >= SPC) return 0u;
    int cx = sc.x;
    int cz = sc.z;
    if (cx < loadMinCX || cx >= loadMaxCX ||
        cz < loadMinCZ || cz >= loadMaxCZ) return 0u;
    uint sx = PosMod(cx, LC);
    uint sz = PosMod(cz, LC);
    return Occupancy.Load(int4(sx, sc.y, sz, 0));
}

uint BHash(int3 p)
{
    uint h = uint(p.x)*374761393u + uint(p.y)*668265263u + uint(p.z)*1274126177u;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    return h;
}

float3 BlockColor(uint bt, int face, int3 bp)
{
    float3 c = bcolors[bt];
    if (bt == 3 && face != 2) c = bcolors[2];
    if (bt == 6 && face != 2) c *= float3(0.90,0.92,0.98);
    float v = float(BHash(bp) & 255u) / 255.0 * 0.08 - 0.04;
    return c * (1.0 + v);
}

bool Trace(float3 ro, float3 rd, float maxT,
           out float outDist, out int outFace, out int3 outVox)
{
    int3   st = int3(rd.x>=0?1:-1, rd.y>=0?1:-1, rd.z>=0?1:-1);
    float3 tD = abs(1.0/rd);

    int3   mp = int3(floor(ro));
    float3 tM = InitTMax(mp, float3(1,1,1), ro, st, tD);

    int3   sc  = SectionOf(mp);
    float3 tDC = SEC_F * tD;
    float3 tMC = InitTMax(sc, SEC_F, ro, st, tD);

    float dist = 0.0;
    int   face = -1;
    bool  hit  = false;
    bool  fineValid = true;
    bool  skipFirst = true;

    [loop] for (int ci = 0; ci < MAX_COARSE; ++ci)
    {
        if (dist > maxT) break;
        if (sc.y <  0   && st.y < 0) break;
        if (sc.y >= SPC && st.y > 0) break;

        if (GetOccupancy(sc) != 0u)
        {
            int3 lo = sc * SEC_I;
            int3 hi = lo + SEC_I - int3(1,1,1);

            if (!fineValid) {
                float3 p = ro + rd * dist;
                mp = clamp(int3(floor(p)), lo, hi);
                tM = InitTMax(mp, float3(1,1,1), ro, st, tD);
                fineValid = true;
            }

            [loop] for (int fi = 0; fi < MAX_FINE; ++fi)
            {
                if (!skipFirst) {
                    uint bt = GetBlock(mp);
                    if (bt != 0u) { hit = true; break; }
                }
                skipFirst = false;

                if (tM.x < tM.y) {
                    if (tM.x < tM.z) { dist=tM.x; tM.x+=tD.x; mp.x+=st.x; face=st.x>0?1:0; }
                    else             { dist=tM.z; tM.z+=tD.z; mp.z+=st.z; face=st.z>0?5:4; }
                } else {
                    if (tM.y < tM.z) { dist=tM.y; tM.y+=tD.y; mp.y+=st.y; face=st.y>0?3:2; }
                    else             { dist=tM.z; tM.z+=tD.z; mp.z+=st.z; face=st.z>0?5:4; }
                }

                if (dist > maxT) break;
                if (any(mp < lo) || any(mp > hi)) break;
            }
            if (hit || dist > maxT) break;

            sc  = SectionOf(mp);
            tMC = InitTMax(sc, SEC_F, ro, st, tD);
        }
        else
        {
            skipFirst = false;
            if (tMC.x < tMC.y) {
                if (tMC.x < tMC.z) { dist=tMC.x; tMC.x+=tDC.x; sc.x+=st.x; face=st.x>0?1:0; }
                else               { dist=tMC.z; tMC.z+=tDC.z; sc.z+=st.z; face=st.z>0?5:4; }
            } else {
                if (tMC.y < tMC.z) { dist=tMC.y; tMC.y+=tDC.y; sc.y+=st.y; face=st.y>0?3:2; }
                else               { dist=tMC.z; tMC.z+=tDC.z; sc.z+=st.z; face=st.z>0?5:4; }
            }
            fineValid = false;
        }
    }

    outDist = dist;
    outFace = face;
    outVox  = mp;
    return hit;
}

[numthreads(8,8,1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= screenW || tid.y >= screenH) return;

    float asp = float(screenW) / float(max(1u, screenH));
    float px  = (2.0*(float(tid.x)+0.5)/float(screenW)-1.0) * asp * tanHalfFov;
    float py  = (1.0-2.0*(float(tid.y)+0.5)/float(screenH)) * tanHalfFov;
    float3 rd = normalize(camFwd + camRight*px + camUp*py);

    float dist; int face; int3 mp;
    bool hit = Trace(camPos, rd, maxDist, dist, face, mp);

    float4 color;
    if (hit) {
        uint bt = GetBlock(mp);
        float3 c = BlockColor(bt, face, mp) * faceBri[face];

        // ---- World-space ambient occlusion (6 fixed secondary rays) ----
        float3 N = faceN[face];
        float3 T = faceT[face];
        float3 B = faceB[face];
        float3 hitPos   = camPos + rd * dist;
        float3 aoOrigin = hitPos + N * AO_BIAS;

        float occ = 0.0;
        [loop] for (int s = 0; s < AO_SAMPLES; ++s)
        {
            float3 h     = AO_DIRS[s];
            float3 aoDir = T * h.x + B * h.y + N * h.z;
            float  aoDist; int aoFace; int3 aoVox;
            if (Trace(aoOrigin, aoDir, AO_MAX_DIST, aoDist, aoFace, aoVox))
                occ += saturate(1.0 - aoDist / AO_MAX_DIST);
        }
        float ao = 1.0 - occ / float(AO_SAMPLES);

        // ---- Directional AO modulation: sun-facing surfaces receive less AO ----
        float NdotL = dot(N, AO_SUN_DIR);
        float sunFacing = saturate(0.5 - 0.5 * NdotL);     // 0 = facing sun, 1 = away
        float dirMod = lerp(1.0, sunFacing, AO_SUN_INFLUENCE);
        float effStrength = AO_STRENGTH * dirMod;
        c *= max(AO_MIN_BRIGHT, (1.0 - effStrength) + effStrength * ao);

        float fogT = saturate(dist/maxDist); fogT *= fogT;
        c = lerp(c, FOG_CLR, fogT);
        color = float4(c, 1.0);
    } else {
        float t = rd.y*0.5+0.5;
        color = float4(lerp(float3(0.67,0.80,0.92), float3(0.25,0.45,0.75), saturate(t)), 1);
    }
    Output[tid.xy] = color;
}
)";

// ===================================================================
//  D3D12 utilities
// ===================================================================
static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r,
    D3D12_RESOURCE_STATES bef, D3D12_RESOURCE_STATES aft) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = bef;
    b.Transition.StateAfter = aft;
    return b;
}

static void WaitForGpu() {
    Check(gpu.queue->Signal(gpu.fence.Get(), gpu.fenceValues[gpu.frameIndex]), "Sig");
    if (gpu.fence->GetCompletedValue() < gpu.fenceValues[gpu.frameIndex]) {
        Check(gpu.fence->SetEventOnCompletion(gpu.fenceValues[gpu.frameIndex],
            gpu.fenceEvent), "Evt");
        WaitForSingleObject(gpu.fenceEvent, INFINITE);
    }
    gpu.fenceValues[gpu.frameIndex]++;
}

static void MoveToNextFrame() {
    UINT64 sub = gpu.fenceValues[gpu.frameIndex];
    Check(gpu.queue->Signal(gpu.fence.Get(), sub), "Sig");
    gpu.frameIndex = gpu.swapChain->GetCurrentBackBufferIndex();
    if (gpu.fence->GetCompletedValue() < gpu.fenceValues[gpu.frameIndex]) {
        Check(gpu.fence->SetEventOnCompletion(gpu.fenceValues[gpu.frameIndex],
            gpu.fenceEvent), "Evt");
        WaitForSingleObject(gpu.fenceEvent, INFINITE);
    }
    gpu.fenceValues[gpu.frameIndex] = sub + 1;
}

static ComPtr<ID3D12Resource> CreateUploadBuffer(UINT64 sz) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sz; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> r;
    Check(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&r)),
        "UploadBuf");
    return r;
}

static ComPtr<ID3D12Resource> CreateDefaultTex2D(DXGI_FORMAT fmt, UINT w, UINT h,
    D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w; rd.Height = h; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.Format = fmt; rd.SampleDesc.Count = 1; rd.Flags = fl;
    ComPtr<ID3D12Resource> r;
    Check(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        st, nullptr, IID_PPV_ARGS(&r)), "Tex2D");
    return r;
}

static ComPtr<ID3D12Resource> CreateDefaultTex3D(DXGI_FORMAT fmt,
    UINT w, UINT h, UINT d,
    D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    rd.Width = w; rd.Height = h;
    rd.DepthOrArraySize = UINT16(d);
    rd.MipLevels = 1; rd.Format = fmt; rd.SampleDesc.Count = 1; rd.Flags = fl;
    ComPtr<ID3D12Resource> r;
    Check(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        st, nullptr, IID_PPV_ARGS(&r)), "Tex3D");
    return r;
}

// Record a CopyTextureRegion for one chunk from staging into the atlas.
static void RecordChunkCopy(ID3D12GraphicsCommandList* cl,
    ID3D12Resource* staging, UINT64 stagingOffset,
    ID3D12Resource* atlas, int cx, int cz) {

    int ax = AtlasSlot(cx) * CHUNK_X;
    int az = AtlasSlot(cz) * CHUNK_Z;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = staging;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = stagingOffset;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8_UINT;
    src.PlacedFootprint.Footprint.Width = CHUNK_X;
    src.PlacedFootprint.Footprint.Height = CHUNK_Y;
    src.PlacedFootprint.Footprint.Depth = CHUNK_Z;
    src.PlacedFootprint.Footprint.RowPitch = STAGING_ROW;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = atlas;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    cl->CopyTextureRegion(&dst, UINT(ax), 0, UINT(az), &src, nullptr);
}

// Record a CopyTextureRegion for one chunk's occupancy column
// (1 × SECTIONS_PER_CHUNK × 1) into the occupancy map.
static void RecordOccCopy(ID3D12GraphicsCommandList* cl,
    ID3D12Resource* staging, UINT64 stagingOffset,
    ID3D12Resource* occ, int cx, int cz) {

    int ax = AtlasSlot(cx);
    int az = AtlasSlot(cz);

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = staging;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = stagingOffset;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8_UINT;
    src.PlacedFootprint.Footprint.Width = 1;
    src.PlacedFootprint.Footprint.Height = SECTIONS_PER_CHUNK;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = OCC_ROW;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = occ;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    cl->CopyTextureRegion(&dst, UINT(ax), 0, UINT(az), &src, nullptr);
}

// ===================================================================
//  Size-dependent resources
// ===================================================================
static void CreateSizeDependentResources() {
    gpu.outputTex = CreateDefaultTex2D(DXGI_FORMAT_R8G8B8A8_UNORM,
        gApp.width, gApp.height,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Heap layout: [0]=atlas SRV, [1]=occupancy SRV, [2]=output UAV
    D3D12_CPU_DESCRIPTOR_HANDLE h = gpu.srvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += 2 * gpu.srvDescSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(gpu.outputTex.Get(), nullptr, &uav, h);

    for (UINT i = 0; i < FRAME_COUNT; ++i)
        Check(gpu.swapChain->GetBuffer(i, IID_PPV_ARGS(&gpu.backBuffers[i])), "BB");
}

void ResizeBackbuffer(int w, int h) {
    if (!gpu.device) { gApp.width = w; gApp.height = h; return; }
    WaitForGpu();
    for (UINT i = 0; i < FRAME_COUNT; ++i) {
        gpu.backBuffers[i].Reset();
        gpu.fenceValues[i] = gpu.fenceValues[gpu.frameIndex];
    }
    gpu.outputTex.Reset();
    gApp.width = std::max(1, w);
    gApp.height = std::max(1, h);
    Check(gpu.swapChain->ResizeBuffers(FRAME_COUNT, gApp.width, gApp.height,
        DXGI_FORMAT_R8G8B8A8_UNORM, 0), "Resize");
    gpu.frameIndex = gpu.swapChain->GetCurrentBackBufferIndex();
    CreateSizeDependentResources();
}

// ===================================================================
//  Upload a batch of chunks (blocking).  Used during init.
//  Uploads both voxel data (atlas) and section occupancy (occupancy map).
// ===================================================================
static void UploadChunkBatch(const std::vector<std::pair<int, int>>& batch) {
    if (batch.empty()) return;

    uint8_t* mapped = gpu.frameStagingMapped[0];
    uint8_t* occMapped = gpu.frameOccStagingMapped[0];
    for (size_t i = 0; i < batch.size(); ++i) {
        auto [cx, cz] = batch[i];
        auto it = cm.chunks.find(ChunkKey(cx, cz));
        FlattenChunk(it->second, mapped + i * CHUNK_STAGING);
        FlattenOccupancy(it->second, occMapped + i * CHUNK_OCC_STAGING);
    }

    Check(gpu.cmdAlloc[0]->Reset(), "AR");
    Check(gpu.cmdList->Reset(gpu.cmdAlloc[0].Get(), nullptr), "LR");

    for (size_t i = 0; i < batch.size(); ++i) {
        auto [cx, cz] = batch[i];
        RecordChunkCopy(gpu.cmdList.Get(), gpu.frameStaging[0].Get(),
            i * CHUNK_STAGING, gpu.atlas.Get(), cx, cz);
        RecordOccCopy(gpu.cmdList.Get(), gpu.frameOccStaging[0].Get(),
            i * CHUNK_OCC_STAGING, gpu.occupancy.Get(), cx, cz);
    }

    Check(gpu.cmdList->Close(), "Cl");
    ID3D12CommandList* ls[] = { gpu.cmdList.Get() };
    gpu.queue->ExecuteCommandLists(1, ls);
    WaitForGpu();
}

// ===================================================================
//  InitD3D12
// ===================================================================
void InitD3D12(HWND hwnd) {
    gApp.hwnd = hwnd;

#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> d;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&d)))) d->EnableDebugLayer();
    }
#endif

    ComPtr<IDXGIFactory4> fac;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&fac)), "Fac");
    Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&gpu.device)), "Dev");

    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Check(gpu.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&gpu.queue)), "Q");

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.BufferCount = FRAME_COUNT; scd.Width = gApp.width; scd.Height = gApp.height;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;
    ComPtr<IDXGISwapChain1> sc1;
    Check(fac->CreateSwapChainForHwnd(gpu.queue.Get(), hwnd, &scd,
        nullptr, nullptr, &sc1), "SC");
    Check(sc1.As(&gpu.swapChain), "SC3");
    gpu.frameIndex = gpu.swapChain->GetCurrentBackBufferIndex();

    // Descriptor heap: [0]=atlas SRV, [1]=occupancy SRV, [2]=output UAV
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.NumDescriptors = 3;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&gpu.srvHeap)), "Hp");
    gpu.srvDescSize = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    for (UINT i = 0; i < FRAME_COUNT; ++i)
        Check(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&gpu.cmdAlloc[i])), "Al");

    Check(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        gpu.cmdAlloc[0].Get(), nullptr, IID_PPV_ARGS(&gpu.cmdList)), "CL");
    Check(gpu.cmdList->Close(), "CLClose");

    Check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&gpu.fence)), "Fn");
    gpu.fenceValues[gpu.frameIndex] = 1;
    gpu.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // ---- Root signature: [0] CBV b0, [1] table {t0, t1, u0} ----
    D3D12_DESCRIPTOR_RANGE rng[2]{};
    rng[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    rng[0].NumDescriptors = 2; rng[0].BaseShaderRegister = 0;           // t0, t1
    rng[0].OffsetInDescriptorsFromTableStart = 0;
    rng[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    rng[1].NumDescriptors = 1; rng[1].BaseShaderRegister = 0;           // u0
    rng[1].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER par[2]{};
    par[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    par[0].Descriptor.ShaderRegister = 0;
    par[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    par[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    par[1].DescriptorTable.NumDescriptorRanges = 2;
    par[1].DescriptorTable.pDescriptorRanges = rng;
    par[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2; rsd.pParameters = par;

    ComPtr<ID3DBlob> sig, err;
    Check(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
        &sig, &err), "SRS");
    Check(gpu.device->CreateRootSignature(0, sig->GetBufferPointer(),
        sig->GetBufferSize(), IID_PPV_ARGS(&gpu.rootSig)), "RS");

    // ---- Compile CS ----
    ComPtr<ID3DBlob> cs;
    HRESULT hr = D3DCompile(g_shaderSrc, strlen(g_shaderSrc), nullptr, nullptr, nullptr,
        "CSMain", "cs_5_1", 0, 0, &cs, &err);
    if (FAILED(hr)) {
        if (err) OutputDebugStringA(static_cast<const char*>(err->GetBufferPointer()));
        Check(hr, "CS");
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = gpu.rootSig.Get();
    psd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    Check(gpu.device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&gpu.pso)), "PSO");

    // ---- CB ring ----
    gpu.cbUpload = CreateUploadBuffer(FRAME_COUNT * CB_ALIGN);
    Check(gpu.cbUpload->Map(0, nullptr, reinterpret_cast<void**>(&gpu.cbMapped)), "MCB");

    // ---- Per-frame staging buffers (voxel + occupancy) ----
    for (UINT i = 0; i < FRAME_COUNT; ++i) {
        gpu.frameStaging[i] = CreateUploadBuffer(FRAME_STAGING_SIZE);
        Check(gpu.frameStaging[i]->Map(0, nullptr,
            reinterpret_cast<void**>(&gpu.frameStagingMapped[i])), "MS");

        gpu.frameOccStaging[i] = CreateUploadBuffer(FRAME_OCC_STAGING_SIZE);
        Check(gpu.frameOccStaging[i]->Map(0, nullptr,
            reinterpret_cast<void**>(&gpu.frameOccStagingMapped[i])), "MOS");
    }

    // ---- Atlas + Occupancy (COPY_DEST initially for bulk upload) ----
    gpu.atlas = CreateDefaultTex3D(DXGI_FORMAT_R8_UINT,
        ATLAS_XZ, CHUNK_Y, ATLAS_XZ,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    gpu.occupancy = CreateDefaultTex3D(DXGI_FORMAT_R8_UINT,
        LOAD_CHUNKS, SECTIONS_PER_CHUNK, LOAD_CHUNKS,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    // ---- Bulk upload all queued chunks (voxels + occupancy) ----
    {
        std::vector<std::pair<int, int>> batch;
        batch.reserve(MAX_UPLOADS_PER_FRAME);
        while (!cm.uploadQueue.empty()) {
            batch.clear();
            for (int i = 0; i < MAX_UPLOADS_PER_FRAME && !cm.uploadQueue.empty(); ++i) {
                batch.push_back(cm.uploadQueue.front());
                cm.uploadQueue.pop_front();
            }
            UploadChunkBatch(batch);
        }
    }

    // Transition atlas + occupancy COPY_DEST -> NON_PIXEL_SHADER_RESOURCE
    {
        Check(gpu.cmdAlloc[0]->Reset(), "AR");
        Check(gpu.cmdList->Reset(gpu.cmdAlloc[0].Get(), nullptr), "LR");
        D3D12_RESOURCE_BARRIER b[2] = {
            Transition(gpu.atlas.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            Transition(gpu.occupancy.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        };
        gpu.cmdList->ResourceBarrier(2, b);
        Check(gpu.cmdList->Close(), "Cl");
        ID3D12CommandList* ls[] = { gpu.cmdList.Get() };
        gpu.queue->ExecuteCommandLists(1, ls);
        WaitForGpu();
    }

    // ---- Atlas SRV (slot 0) ----
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h = gpu.srvHeap->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = DXGI_FORMAT_R8_UINT;
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sv.Texture3D.MipLevels = 1;
        gpu.device->CreateShaderResourceView(gpu.atlas.Get(), &sv, h);
    }

    // ---- Occupancy SRV (slot 1) ----
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h = gpu.srvHeap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += gpu.srvDescSize;
        D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = DXGI_FORMAT_R8_UINT;
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sv.Texture3D.MipLevels = 1;
        gpu.device->CreateShaderResourceView(gpu.occupancy.Get(), &sv, h);
    }

    // ---- Zero staging for atlas clears (one chunk, zeroed once) ----
    gpu.zeroStaging = CreateUploadBuffer(CHUNK_STAGING);
    {
        uint8_t* zp = nullptr;
        Check(gpu.zeroStaging->Map(0, nullptr, reinterpret_cast<void**>(&zp)), "MZ");
        std::memset(zp, 0, CHUNK_STAGING);
        gpu.zeroStaging->Unmap(0, nullptr);
    }

    // ---- Zero staging for occupancy clears (one column, zeroed once) ----
    gpu.zeroOccStaging = CreateUploadBuffer(CHUNK_OCC_STAGING);
    {
        uint8_t* zp = nullptr;
        Check(gpu.zeroOccStaging->Map(0, nullptr, reinterpret_cast<void**>(&zp)), "MZO");
        std::memset(zp, 0, CHUNK_OCC_STAGING);
        gpu.zeroOccStaging->Unmap(0, nullptr);
    }

    CreateSizeDependentResources();
}

// ===================================================================
//  Render  (streaming uploads + dispatch + present)
// ===================================================================
void Render() {
    // ---- Region boundary check ----
    int prx = BlockToRegion(int(std::floor(gApp.camX)));
    int prz = BlockToRegion(int(std::floor(gApp.camY)));
    if (prx != cm.centerRX || prz != cm.centerRZ)
        ScheduleRegionTransition(prx, prz);

    // ---- Consume finished chunks from worker threads (cheap, no noise) ----
    ProcessGenResults();

    auto* alloc = gpu.cmdAlloc[gpu.frameIndex].Get();
    Check(alloc->Reset(), "AR");
    Check(gpu.cmdList->Reset(alloc, gpu.pso.Get()), "LR");

    // ---- Per-frame chunk uploads (skip stale entries) ----
    std::vector<std::pair<int, int>> batch;
    {
        uint8_t* mapped = gpu.frameStagingMapped[gpu.frameIndex];
        uint8_t* occMapped = gpu.frameOccStagingMapped[gpu.frameIndex];
        int uploaded = 0;
        while (uploaded < MAX_UPLOADS_PER_FRAME && !cm.uploadQueue.empty()) {
            auto [cx, cz] = cm.uploadQueue.front();
            cm.uploadQueue.pop_front();
            auto it = cm.chunks.find(ChunkKey(cx, cz));
            if (it == cm.chunks.end()) continue;
            FlattenChunk(it->second, mapped + size_t(uploaded) * CHUNK_STAGING);
            FlattenOccupancy(it->second, occMapped + size_t(uploaded) * CHUNK_OCC_STAGING);
            batch.push_back({ cx, cz });
            ++uploaded;
        }
    }

    bool hasAtlasWork = !cm.occClearQueue.empty()
        || !cm.atlasClearQueue.empty()
        || !batch.empty();

    if (hasAtlasWork) {
        D3D12_RESOURCE_BARRIER bPre[2] = {
            Transition(gpu.atlas.Get(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST),
            Transition(gpu.occupancy.Get(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST)
        };
        gpu.cmdList->ResourceBarrier(2, bPre);

        // --- Occupancy clears: ALL of them this frame.
        //     Each is a 1×8×1 copy (8 effective bytes); zeroing occupancy
        //     immediately makes the stale atlas voxels unreachable by the
        //     ray marcher, so no stale geometry is ever drawn.
        while (!cm.occClearQueue.empty()) {
            auto [cx, cz] = cm.occClearQueue.front();
            cm.occClearQueue.pop_front();
            int sx = AtlasSlot(cx), sz = AtlasSlot(cz);
            if (cm.slotKeys[sx][sz] != INT64_MIN) continue;      // slot already re-claimed
            RecordOccCopy(gpu.cmdList.Get(), gpu.zeroOccStaging.Get(), 0,
                gpu.occupancy.Get(), cx, cz);
        }

        // --- Atlas voxel clears: amortised across frames.
        //     Skip any slot that has since been re-claimed so we never
        //     clobber freshly uploaded data.
        {
            int cleared = 0;
            while (cleared < MAX_CLEARS_PER_FRAME && !cm.atlasClearQueue.empty()) {
                auto [cx, cz] = cm.atlasClearQueue.front();
                cm.atlasClearQueue.pop_front();
                int sx = AtlasSlot(cx), sz = AtlasSlot(cz);
                if (cm.slotKeys[sx][sz] != INT64_MIN) continue;  // stale request – discard
                RecordChunkCopy(gpu.cmdList.Get(), gpu.zeroStaging.Get(), 0,
                    gpu.atlas.Get(), cx, cz);
                ++cleared;
            }
        }

        // --- Uploads: new chunk data + its occupancy column ---
        for (size_t i = 0; i < batch.size(); ++i) {
            RecordChunkCopy(gpu.cmdList.Get(),
                gpu.frameStaging[gpu.frameIndex].Get(),
                UINT64(i) * CHUNK_STAGING,
                gpu.atlas.Get(), batch[i].first, batch[i].second);
            RecordOccCopy(gpu.cmdList.Get(),
                gpu.frameOccStaging[gpu.frameIndex].Get(),
                UINT64(i) * CHUNK_OCC_STAGING,
                gpu.occupancy.Get(), batch[i].first, batch[i].second);
        }

        D3D12_RESOURCE_BARRIER bPost[2] = {
            Transition(gpu.atlas.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            Transition(gpu.occupancy.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        };
        gpu.cmdList->ResourceBarrier(2, bPost);
    }

    // ---- Frame constants (unchanged from previous version) ----
    float cosP = std::cos(gApp.pitch), sinP = std::sin(gApp.pitch);
    float cosA = std::cos(gApp.angle), sinA = std::sin(gApp.angle);

    FrameConstants fc{};
    fc.camPos[0] = gApp.camX;
    fc.camPos[1] = gApp.camZ;
    fc.camPos[2] = gApp.camY;
    fc.fwd[0] = cosP * cosA;  fc.fwd[1] = sinP;  fc.fwd[2] = cosP * sinA;
    fc.right[0] = -sinA;        fc.right[1] = 0.0f;  fc.right[2] = cosA;
    fc.up[0] = -sinP * cosA; fc.up[1] = cosP;  fc.up[2] = -sinP * sinA;
    fc.screenW = UINT(gApp.width);
    fc.screenH = UINT(gApp.height);
    fc.loadMinCX = (cm.centerRX - 1) * REGION_CHUNKS;
    fc.loadMinCZ = (cm.centerRZ - 1) * REGION_CHUNKS;
    fc.loadMaxCX = (cm.centerRX + 2) * REGION_CHUNKS;
    fc.loadMaxCZ = (cm.centerRZ + 2) * REGION_CHUNKS;
    fc.tanHalfFov = 0.8f;
    fc.maxDist = 300.0f;
    std::memcpy(gpu.cbMapped + gpu.frameIndex * CB_ALIGN, &fc, sizeof(fc));

    // ---- Dispatch + present (identical to previous) ----
    gpu.cmdList->SetComputeRootSignature(gpu.rootSig.Get());
    ID3D12DescriptorHeap* heaps[] = { gpu.srvHeap.Get() };
    gpu.cmdList->SetDescriptorHeaps(1, heaps);
    gpu.cmdList->SetComputeRootConstantBufferView(0,
        gpu.cbUpload->GetGPUVirtualAddress() + gpu.frameIndex * CB_ALIGN);
    gpu.cmdList->SetComputeRootDescriptorTable(1,
        gpu.srvHeap->GetGPUDescriptorHandleForHeapStart());

    gpu.cmdList->Dispatch((gApp.width + 7) / 8, (gApp.height + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER pre[2] = {
        Transition(gpu.outputTex.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        Transition(gpu.backBuffers[gpu.frameIndex].Get(),
            D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST)
    };
    gpu.cmdList->ResourceBarrier(2, pre);
    gpu.cmdList->CopyResource(gpu.backBuffers[gpu.frameIndex].Get(), gpu.outputTex.Get());

    D3D12_RESOURCE_BARRIER post[2] = {
        Transition(gpu.outputTex.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        Transition(gpu.backBuffers[gpu.frameIndex].Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)
    };
    gpu.cmdList->ResourceBarrier(2, post);

    Check(gpu.cmdList->Close(), "Cl");
    ID3D12CommandList* ls[] = { gpu.cmdList.Get() };
    gpu.queue->ExecuteCommandLists(1, ls);
    Check(gpu.swapChain->Present(1, 0), "Pr");
    MoveToNextFrame();
}

// ===================================================================
//  Shutdown
// ===================================================================
    void ShutdownD3D12() {
        StopGenWorkers();                 // join workers first – safe even if never started
        if (!gpu.device) return;
        WaitForGpu();
        for (UINT i = 0; i < FRAME_COUNT; ++i) {
            if (gpu.frameStagingMapped[i]) {
                gpu.frameStaging[i]->Unmap(0, nullptr);
                gpu.frameStagingMapped[i] = nullptr;
            }
            if (gpu.frameOccStagingMapped[i]) {
                gpu.frameOccStaging[i]->Unmap(0, nullptr);
                gpu.frameOccStagingMapped[i] = nullptr;
            }
        }
        if (gpu.cbMapped) { gpu.cbUpload->Unmap(0, nullptr); gpu.cbMapped = nullptr; }
        if (gpu.fenceEvent) { CloseHandle(gpu.fenceEvent);     gpu.fenceEvent = nullptr; }
        cm.chunks.clear();
    }

// ===================================================================
//  Camera / input
// ===================================================================
static bool KeyDown(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

static void CenterCursor() {
    RECT r; GetClientRect(gApp.hwnd, &r);
    POINT c = { (r.right - r.left) / 2, (r.bottom - r.top) / 2 };
    ClientToScreen(gApp.hwnd, &c);
    SetCursorPos(c.x, c.y);
}

void UpdateCamera(float dt) {
    float moveSpeed = 8.0f, strafeSpeed = 7.0f, liftSpeed = 8.0f;
    if (KeyDown(VK_SHIFT)) { moveSpeed *= 3; strafeSpeed *= 3; liftSpeed *= 3; }

    if (gApp.mouseCaptured) {
        RECT cr; GetClientRect(gApp.hwnd, &cr);
        POINT sc = { (cr.right - cr.left) / 2, (cr.bottom - cr.top) / 2 };
        ClientToScreen(gApp.hwnd, &sc);
        POINT cur; GetCursorPos(&cur);
        float dx = float(cur.x - sc.x), dy = float(cur.y - sc.y);
        gApp.angle += dx * 0.002f;
        gApp.pitch -= dy * 0.002f;
        gApp.pitch = std::clamp(gApp.pitch, -1.48f, 1.48f);
        SetCursorPos(sc.x, sc.y);
    }
    if (KeyDown(VK_LEFT))  gApp.angle -= 1.7f * dt;
    if (KeyDown(VK_RIGHT)) gApp.angle += 1.7f * dt;

    float fwd = 0, str = 0;
    if (KeyDown('W') || KeyDown(VK_UP))   fwd += 1;
    if (KeyDown('S') || KeyDown(VK_DOWN)) fwd -= 1;
    if (KeyDown('A') || KeyDown('Q'))     str -= 1;
    if (KeyDown('D') || KeyDown('E'))     str += 1;

    float cosA = std::cos(gApp.angle), sinA = std::sin(gApp.angle);
    gApp.camX += cosA * fwd * moveSpeed * dt;
    gApp.camY += sinA * fwd * moveSpeed * dt;
    gApp.camX += -sinA * str * strafeSpeed * dt;
    gApp.camY += cosA * str * strafeSpeed * dt;

    if (KeyDown('T')) gApp.flyMode = true;
    if (KeyDown('G')) gApp.flyMode = false;

    if (gApp.flyMode) {
        if (KeyDown(VK_SPACE) || KeyDown('R')) gApp.camZ += liftSpeed * dt;
        if (KeyDown(VK_CONTROL) || KeyDown('F')) gApp.camZ -= liftSpeed * dt;
    }
    else {
        float sh = float(SurfaceHeightAt(gApp.camX, gApp.camY));
        float target = sh + 1.0f + 1.5f;
        float follow = std::clamp(dt * 8.0f, 0.0f, 1.0f);
        gApp.camZ = Lerp(gApp.camZ, target, follow);
    }
}

void UpdateWindowTitle(HWND hwnd, float dt) {
    gApp.fpsTimer += dt; gApp.fpsFrames++;
    if (gApp.fpsTimer >= 0.5f) {
        float fps = float(gApp.fpsFrames) / gApp.fpsTimer;
        int loaded = int(cm.chunks.size());
        int queued = int(cm.uploadQueue.size());
        wchar_t buf[256];
        swprintf_s(buf, L"Voxel Terrain | chunks:%d queue:%d | FPS:%d", loaded, queued, int(fps));
        SetWindowTextW(hwnd, buf);
        gApp.fpsTimer = 0; gApp.fpsFrames = 0;
    }
}

// ===================================================================
//  Window procedure
// ===================================================================
LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_SIZE: {
        int w = LOWORD(lParam), h = HIWORD(lParam);
        if (w > 0 && h > 0) ResizeBackbuffer(w, h);
        return 0;
    }
    case WM_LBUTTONDOWN:
        if (!gApp.mouseCaptured) {
            gApp.mouseCaptured = true;
            ShowCursor(FALSE);
            CenterCursor();
        }
        return 0;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE && gApp.mouseCaptured) {
            gApp.mouseCaptured = false;
            ShowCursor(TRUE);
            return 0;
        }
        break;
    case WM_KILLFOCUS:
        if (gApp.mouseCaptured) {
            gApp.mouseCaptured = false;
            ShowCursor(TRUE);
        }
        return 0;
    case WM_CLOSE:   DestroyWindow(hwnd); return 0;
    case WM_DESTROY: gApp.running = false; PostQuitMessage(0); return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}