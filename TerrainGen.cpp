#include "TerrainGen.h"

#include "TerrainNoise.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

// ===================================================================
//  Region surface-height cache
//
//  A region is REGION_CHUNKS × REGION_CHUNKS chunks = 256×256 columns.
//  One HeightTile stores the *final clamped* surface height for every
//  column in a region.  The tile is computed lazily, exactly once per
//  generation epoch, by whichever worker asks for it first; every
//  other chunk in that region (on any worker thread) then reads the
//  same immutable tile instead of re-running the FBM stack.
//
//  Because the height formula is a pure function of world (wx, wz)
//  and is reproduced verbatim in ComputeRegionHeights, a column on
//  the edge of region R evaluates to the IDENTICAL value as the same
//  world column evaluated from region R±1 — so region seams remain
//  perfectly continuous.  Cross-region sampling is simply
//  GetRegionHeights(neighbourRx, neighbourRz).
// ===================================================================
namespace {

    static constexpr int WATER_LEVEL = 50;
    static constexpr int REGION_DIM = REGION_CHUNKS * CHUNK_X; // 256

    struct HeightTile {
        std::atomic<int> nextRow{ 0 };
        std::atomic<int> doneRows{ 0 };

        uint8_t h[REGION_DIM * REGION_DIM];

        bool ready() const {
            return doneRows.load(std::memory_order_acquire) >= REGION_DIM;
        }

        uint8_t at(int lx, int lz) const {
            return h[lz * REGION_DIM + lx];
        }
    };

    int64_t RegionKey(int rx, int rz) {
        return (int64_t(rx) << 32) | (int64_t(rz) & 0xFFFFFFFF);
    }

    struct HeightCache {
        std::mutex mtx;
        std::unordered_map<int64_t, std::shared_ptr<HeightTile>> tiles;
        uint32_t epoch = 0;
    } hcache;

// -------------------------------------------------------------------
//  Cooperative region fill.  Every worker that needs this tile calls
//  in here and repeatedly claims the next free row via fetch_add, so
//  N workers split the 256 rows ~N ways with no idle blocking.  The
//  per-column formula is *unchanged* from the original inline code.
// -------------------------------------------------------------------

    void FillRegionHeights(int rx, int rz, HeightTile& t) {
        const int wx0 = rx * REGION_DIM;
        const int wz0 = rz * REGION_DIM;

        int lz;
        while ((lz = t.nextRow.fetch_add(1, std::memory_order_relaxed)) < REGION_DIM) {
            const int wz = wz0 + lz;
            uint8_t* row = &t.h[size_t(lz) * REGION_DIM];

            for (int lx = 0; lx < REGION_DIM; ++lx) {
                const int wx = wx0 + lx;

                float large = FBM(wx * 0.0012f, wz * 0.0012f, 5);
                float medium = FBM(wx * 0.0035f, wz * 0.0035f, 6);
                float detail = FBM(wx * 0.012f, wz * 0.012f, 4);

                float e = 0.55f * medium + 0.30f * large + 0.15f * detail;
                e = std::pow(e, 1.35f);

                float ridge = std::fabs(FBM(wx * 0.006f, wz * 0.006f, 5) - 0.5f) * 2.0f;

                int surfH = std::clamp(
                    int(20.0f + e * 170.0f + ridge * 18.0f),
                    0,
                    CHUNK_Y - 1
                );

                row[lx] = uint8_t(surfH);
            }

            t.doneRows.fetch_add(1, std::memory_order_release);
        }

        while (t.doneRows.load(std::memory_order_acquire) < REGION_DIM) {
            std::this_thread::yield();
        }
    }

// -------------------------------------------------------------------
//  Batching helper.  Thread-safe, lazily-populated accessor that
//  returns the shared, immutable height tile for (rx, rz).
//
//  jobEpoch is the epoch stamped on the *job* that is asking, so a
//  stale in-flight job can be detected here and short-circuited
//  (nullptr) instead of doing – or worse, flushing – work on behalf
//  of an epoch it no longer belongs to.
// -------------------------------------------------------------------
    std::shared_ptr<const HeightTile> GetRegionHeights(int rx, int rz, uint32_t jobEpoch) {
        const int64_t key = RegionKey(rx, rz);

        std::shared_ptr<HeightTile> tile;
        {
            std::lock_guard<std::mutex> lk(hcache.mtx);

            const int32_t d = int32_t(jobEpoch - hcache.epoch);

            if (d < 0) {
                return nullptr;
            }

            if (d > 0) {
                hcache.tiles.clear();
                hcache.epoch = jobEpoch;
            }

            auto& slot = hcache.tiles[key];
            if (!slot) {
                slot = std::make_shared<HeightTile>();
            }

            tile = slot;
        }

        if (!tile->ready()) {
            FillRegionHeights(rx, rz, *tile);
        }

        return tile;
    }

} // namespace

// ===================================================================
//  Per-chunk terrain generation – PURE.  Builds and returns a Chunk
//  without touching any shared state so it is safe to call from any
//  worker thread concurrently.
// ===================================================================
std::unique_ptr<Chunk> BuildChunkTerrain(int cx, int cz, uint32_t jobEpoch) {
    const int rx = ChunkToRegion(cx);
    const int rz = ChunkToRegion(cz);

    const auto tile = GetRegionHeights(rx, rz, jobEpoch);
    if (!tile) {
        return nullptr;
    }

    auto chunk = std::make_unique<Chunk>();

    const int ox = (cx - rx * REGION_CHUNKS) * CHUNK_X;
    const int oz = (cz - rz * REGION_CHUNKS) * CHUNK_Z;

    for (int lz = 0; lz < CHUNK_Z; ++lz) {
        for (int lx = 0; lx < CHUNK_X; ++lx) {
            const int surfH = tile->at(ox + lx, oz + lz);

            chunk->surfaceY[lz * CHUNK_X + lx] = uint8_t(surfH);

            BlockType surfType;
            if (surfH < 58) {
                surfType = BLOCK_SAND;
            }
            else if (surfH < 140) {
                surfType = BLOCK_GRASS;
            }
            else if (surfH < 180) {
                surfType = BLOCK_ROCK;
            }
            else {
                surfType = BLOCK_SNOW;
            }

            for (int y = 0; y <= surfH; ++y) {
                BlockType bt;

                if (y < surfH - 4) {
                    bt = BLOCK_STONE;
                }
                else if (y < surfH) {
                    bt = BLOCK_DIRT;
                }
                else {
                    bt = surfType;
                }

                chunk->setBlock(lx, y, lz, bt);
            }

            for (int y = surfH + 1; y <= WATER_LEVEL && y < CHUNK_Y; ++y) {
                chunk->setBlock(
                    lx,
                    y,
                    lz,
                    y < 42 ? BLOCK_DEEP_WATER : BLOCK_WATER
                );
            }
        }
    }

    return chunk;
}