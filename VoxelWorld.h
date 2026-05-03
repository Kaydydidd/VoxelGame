#pragma once

#include "voxel.h"

#include <cstdint>
#include <memory>

struct ChunkSection {
    uint8_t  blocks[CHUNK_X * SECTION_Y * CHUNK_Z]{};
    uint16_t solidCount = 0;
};

struct Chunk {
    std::unique_ptr<ChunkSection> sections[SECTIONS_PER_CHUNK];
    uint8_t surfaceY[CHUNK_X * CHUNK_Z]{};

    uint8_t getBlock(int lx, int ly, int lz) const;
    void setBlock(int lx, int ly, int lz, uint8_t bt);
};

int64_t ChunkKey(int cx, int cz);
int AtlasSlot(int c);
int ChunkToRegion(int c);