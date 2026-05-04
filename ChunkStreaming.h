// ChunkStreaming.h

#pragma once

#include "VoxelWorld.h"

#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <utility>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include <memory>

struct ChunkMgr {
    // ---- chunk manager state ----
    std::unordered_map<int64_t, Chunk> chunks;

    int centerRX = std::numeric_limits<int>::min();
    int centerRZ = std::numeric_limits<int>::min();

    int64_t slotKeys[LOAD_CHUNKS][LOAD_CHUNKS];

    std::deque<std::pair<int, int>> uploadQueue;

    // Split clear path:
    // occClearQueue   = tiny occupancy clears
    // atlasClearQueue = full voxel atlas clears, amortized
    std::deque<std::pair<int, int>> occClearQueue;
    std::deque<std::pair<int, int>> atlasClearQueue;

    void init();
};

extern ChunkMgr cm;

// ===================================================================
//  Chunk data structures (file-local)
// ===================================================================

//  Background terrain-generation worker pool
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
};

extern GenWorkerPool gen;

uint8_t SurfaceHeightAt(float wx, float wz);

int LoadedChunkCount();
int UploadQueueCount();

void EvictRegion(int rx, int rz);
void StartGenWorkers(int n);
void StopGenWorkers();
void ProcessGenResults();
void ScheduleRegionTransition(int newRX, int newRZ);