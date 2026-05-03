#include "ChunkStreaming.h"

#include <cstdint>
#include <limits>
#include <cmath>

ChunkMgr cm;

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

// ===================================================================
//  Camera and Window helpers
// ===================================================================

int LoadedChunkCount() {
    return int(cm.chunks.size());
}

int UploadQueueCount() {
    return int(cm.uploadQueue.size());
}