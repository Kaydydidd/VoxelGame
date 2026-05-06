// VoxelWorld.cpp
#include "VoxelWorld.h"

// ===================================================================
//  Chunk data structures (file-local)
// ===================================================================

uint8_t Chunk::getBlock(int lx, int ly, int lz) const {
    if (ly < 0 || ly >= CHUNK_Y) return BLOCK_AIR;

    int si = ly >> 4;

    if (!sections[si]) return BLOCK_AIR;

    return sections[si]->blocks[
        lx + (ly & 15) * CHUNK_X
            + lz * SECTION_Y * CHUNK_X
    ];
}

void Chunk::setBlock(int lx, int ly, int lz, uint8_t bt) {
    if (ly < 0 || ly >= CHUNK_Y) return;

    int si = ly >> 4;
    int idx =
        lx + (ly & 15) * CHUNK_X
        + lz * SECTION_Y * CHUNK_X;

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

    if (s.solidCount == 0) {
        sections[si].reset();
    }
}

int64_t ChunkKey(int cx, int cz) {
    return (int64_t(cx) << 32) | (int64_t(cz) & 0xFFFFFFFF);
}

int AtlasSlot(int c) {
    return ((c % LOAD_CHUNKS) + LOAD_CHUNKS) % LOAD_CHUNKS;
}

int ChunkToRegion(int c) {
    return c >> 3;
}

// ===================================================================
//  Chunk streaming – determine which chunks to load/unload
// ===================================================================
namespace {
    constexpr int REGION_BLOCK_SHIFT = 8; // log2(REGION_CHUNKS * CHUNK_X) = log2(256)
}

int BlockToRegion(int b) {
    return b >> REGION_BLOCK_SHIFT;
}

bool RegionInBounds(
    int rx,
    int rz,
    int centerRX,
    int centerRZ
) {
    return rx >= centerRX - 1 && rx <= centerRX + 1 &&
        rz >= centerRZ - 1 && rz <= centerRZ + 1;
}