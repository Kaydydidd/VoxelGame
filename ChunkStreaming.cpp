#include "ChunkStreaming.h"

#include <cstdint>
#include <limits>

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