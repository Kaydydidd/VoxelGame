// ChunkStreaming.h

#pragma once

#include "VoxelWorld.h"

#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <utility>

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