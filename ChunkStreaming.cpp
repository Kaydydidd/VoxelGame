#include "ChunkStreaming.h"
#include "TerrainGen.h"

#include <cstdint>
#include <limits>
#include <climits>
#include <cmath>
#include <mutex>
#include <utility>
#include <vector>
#include <algorithm>

ChunkMgr cm;
GenWorkerPool gen;

namespace {
    constexpr int GEN_WORKER_COUNT = 2;
}

void ChunkMgr::init() {
    centerRX = std::numeric_limits<int>::min();
    centerRZ = std::numeric_limits<int>::min();

    for (auto& row : slotKeys) {
        for (auto& key : row) {
            key = std::numeric_limits<int64_t>::min();
        }
    }
}

// ===================================================================
//  Surface height query (for camera ground-follow)
// ===================================================================
uint8_t SurfaceHeightAt(float wx, float wz) {
    int bx = int(std::floor(wx));
    int bz = int(std::floor(wz));

    int cx = bx >> 5;
    int cz = bz >> 5;

    int lx = bx & 31;
    int lz = bz & 31;

    auto it = cm.chunks.find(ChunkKey(cx, cz));
    if (it == cm.chunks.end()) {
        return 0;
    }

    return it->second.surfaceY[lz * CHUNK_X + lx];
}

void EvictRegion(int rx, int rz) {
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

// ===================================================================
//  Camera and Window helpers
// ===================================================================

int LoadedChunkCount() {
    return int(cm.chunks.size());
}

int UploadQueueCount() {
    return int(cm.uploadQueue.size());
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

void StartGenWorkers(int n) {
    if (!gen.threads.empty()) return;          // idempotent
    gen.stop = false;
    gen.threads.reserve(size_t(n));
    for (int i = 0; i < n; ++i)
        gen.threads.emplace_back(GenWorkerLoop);
}

void StopGenWorkers() {
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
//  Drain the worker completion queue on the MAIN thread.  This is the
//  only place that inserts into cm.chunks / slotKeys / uploadQueue, so
//  all GPU-facing state mutation remains single-threaded.
// ===================================================================
void ProcessGenResults() {
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


void ScheduleRegionTransition(int newRX, int newRZ) {
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