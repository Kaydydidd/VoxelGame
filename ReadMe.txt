Uhhhhhhh don't worry about there being no folder structure. That's just git. The local project has a folder structure.

I need to implement this file structure and reorganize everything for reals, but this is what I have in VS.


VoxelGame/
├─ README.md
├─ VoxelGame.sln
├─ VoxelGame.vcxproj
├─ VoxelGame.vcxproj.filters
│
└─ Src/
   ├─ Main/
   │  ├─ Demo.cpp
   │  └─ AppState.cpp
   │
   ├─ CameraAndWindow/
   │  ├─ InputCamera.cpp
   │  └─ Window.cpp
   │
   ├─ Rendering/
   │  ├─ voxel.cpp
   │  │
   │  ├─ D3D12/
   │  │  ├─ D3D12ResourceUtils.cpp
   │  │  ├─ D3D12ResourceUtils.h
   │  │  ├─ VoxelUploadCopies.cpp
   │  │  └─ VoxelUploadCopies.h
   │  │
   │  └─ Shaders/
   │     ├─ VoxelShader.cpp
   │     └─ VoxelShader.h
   │
   └─ World/
      ├─ VoxelWorld.cpp
      ├─ VoxelWorld.h
      │
      ├─ Chunking/
      │  ├─ ChunkStreaming.cpp
      │  ├─ ChunkStreaming.h
      │  ├─ VoxelStaging.cpp
      │  └─ VoxelStaging.h
      │
      └─ Terrain/
         ├─ TerrainGen.cpp
         ├─ TerrainGen.h
         ├─ TerrainNoise.cpp
         └─ TerrainNoise.h
