// TerrainGen.h

#pragma once

#include "VoxelWorld.h"

#include <cstdint>
#include <memory>

std::unique_ptr<Chunk> BuildChunkTerrain(int cx, int cz, uint32_t jobEpoch);