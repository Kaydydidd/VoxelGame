// VoxelStaging.h

#pragma once

#include "VoxelWorld.h"

#include <cstdint>

inline constexpr unsigned int STAGING_ROW =
(CHUNK_X + 255u) & ~255u;

inline constexpr unsigned int STAGING_SLICE =
STAGING_ROW * CHUNK_Y;

inline constexpr unsigned int CHUNK_STAGING =
STAGING_SLICE * CHUNK_Z;

inline constexpr unsigned int OCC_ROW = 256;

inline constexpr unsigned int CHUNK_OCC_STAGING =
OCC_ROW * SECTIONS_PER_CHUNK;

void FlattenChunk(const Chunk& c, uint8_t* dst);
void FlattenOccupancy(const Chunk& c, uint8_t* dst);