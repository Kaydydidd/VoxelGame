Desired Generation Pipeline:


initialize renderer, seed, and chunk manager -> 

generate initial player region (chunk size of 32 × 32 × 128, region size of 8 x 8 chunks) -> 

generate chunks within render distance while keeping the player region cached (As in the current 8x8 region the player is in should always be cached at full LOD in order to quickly render them, as well as the 8 surrounding regions.) -> 

build chunk data for each needed chunk -> 

per-chunk upload to GPU -> 

enter frame loop and calculate loaded chunks via player position -> 

queue new chunks on CPU -> 

upload new chunks on GPU (per chunk)




NOTES:
project/
│
├── voxel.h                          ← existing; keep public API + AppState here
│
├── core/
│   ├── Constants.h                  ← all static constexpr values
│   └── FrameConstants.h             ← FrameConstants struct (HLSL CB mirror)
│
├── terrain/
│   ├── Noise.h
│   ├── Noise.cpp
│   ├── Chunk.h
│   ├── Chunk.cpp
│   ├── HeightCache.h
│   ├── HeightCache.cpp
│   ├── TerrainGen.h
│   └── TerrainGen.cpp
│
├── streaming/
│   ├── GenWorker.h
│   ├── GenWorker.cpp
│   ├── ChunkManager.h
│   └── ChunkManager.cpp
│
├── render/
│   ├── GpuState.h                   ← GpuState struct + Check() helper
│   ├── D3D12Helpers.h
│   ├── D3D12Helpers.cpp
│   ├── Atlas.h
│   ├── Atlas.cpp
│   ├── Shader.h                     ← g_shaderSrc string (or reference to .hlsl)
│   ├── D3D12Init.h
│   ├── D3D12Init.cpp
│   ├── Render.h
│   └── Render.cpp
│
└── app/
    ├── Camera.h
    ├── Camera.cpp
    ├── Window.h
    └── Window.cpp

I need to implement this file structure and reorganize everything.