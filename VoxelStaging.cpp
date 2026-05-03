// VoxelStaging.cpp

#include "VoxelStaging.h"

#include <cstring>


// ===================================================================
//  Flatten sparse chunk into staging-buffer layout
// ===================================================================
void FlattenChunk(const Chunk& c, uint8_t* dst) {
    std::memset(dst, BLOCK_AIR, CHUNK_STAGING);

    for (int si = 0; si < SECTIONS_PER_CHUNK; ++si) {
        if (!c.sections[si]) {
            continue;
        }

        const auto& sec = *c.sections[si];

        for (int lz = 0; lz < CHUNK_Z; ++lz) {
            for (int lyl = 0; lyl < SECTION_Y; ++lyl) {
                int wy = si * SECTION_Y + lyl;

                std::memcpy(
                    dst
                    + size_t(lz) * STAGING_SLICE
                    + size_t(wy) * STAGING_ROW,
                    &sec.blocks[
                        lyl * CHUNK_X
                            + lz * SECTION_Y * CHUNK_X
                    ],
                    CHUNK_X
                );
            }
        }
    }
}

// ===================================================================
//  Flatten chunk section-occupancy into staging-buffer layout.
//  Footprint is 1 × SECTIONS_PER_CHUNK × 1 with RowPitch = OCC_ROW, so
//  section si lives at byte offset si * OCC_ROW.
// ===================================================================
void FlattenOccupancy(const Chunk& c, uint8_t* dst) {
    std::memset(dst, 0, CHUNK_OCC_STAGING);

    for (int si = 0; si < SECTIONS_PER_CHUNK; ++si) {
        dst[size_t(si) * OCC_ROW] = c.sections[si] ? 1u : 0u;
    }
}