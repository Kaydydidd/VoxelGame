#include "voxel.h"
#include "VoxelInternal.h"
#include "VoxelShader.h"
#include "TerrainNoise.h"
#include "VoxelWorld.h"
#include "TerrainGen.h"
#include "VoxelStaging.h"

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

// ===================================================================
//  Internal constants
// ===================================================================
static constexpr UINT  FRAME_COUNT = 2;
static constexpr UINT  CB_ALIGN = 256;

static constexpr int   MAX_UPLOADS_PER_FRAME = 8;
static constexpr UINT  FRAME_STAGING_SIZE = MAX_UPLOADS_PER_FRAME * CHUNK_STAGING;
static constexpr int   GEN_WORKER_COUNT = 2;   // number of terrain-gen worker threads (adjustable)
static constexpr int   MAX_CLEARS_PER_FRAME = 32;  // atlas voxel-data clears recorded per frame

// ---- Occupancy-map staging (one 1×SECTIONS_PER_CHUNK×1 column per chunk) ----
static constexpr UINT  FRAME_OCC_STAGING_SIZE = MAX_UPLOADS_PER_FRAME * CHUNK_OCC_STAGING;

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

        if (job.epoch != gen.epoch.load(std::memory_order_relaxed))
            continue;

        std::unique_ptr<Chunk> c = BuildChunkTerrain(job.cx, job.cz, job.epoch);
        if (!c)                                         // height cache reported stale
            continue;

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
 //  Camera and Window helpers
 // ===================================================================
    float GetSurfaceHeight(float wx, float wz) {
        return float(SurfaceHeightAt(wx, wz));
    }

    int GetLoadedChunkCount() {
        return int(cm.chunks.size());
    }

    int GetUploadQueueCount() {
        return int(cm.uploadQueue.size());
    }