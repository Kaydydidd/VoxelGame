#include "VoxelUploadCopies.h"

#include "VoxelWorld.h"
#include "VoxelStaging.h"

// Record a CopyTextureRegion for one chunk from staging into the atlas.
void RecordChunkCopy(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* staging,
    UINT64 stagingOffset,
    ID3D12Resource* atlas,
    int cx,
    int cz
) {
    int ax = AtlasSlot(cx) * CHUNK_X;
    int az = AtlasSlot(cz) * CHUNK_Z;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = staging;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = stagingOffset;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8_UINT;
    src.PlacedFootprint.Footprint.Width = CHUNK_X;
    src.PlacedFootprint.Footprint.Height = CHUNK_Y;
    src.PlacedFootprint.Footprint.Depth = CHUNK_Z;
    src.PlacedFootprint.Footprint.RowPitch = STAGING_ROW;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = atlas;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    commandList->CopyTextureRegion(
        &dst,
        UINT(ax),
        0,
        UINT(az),
        &src,
        nullptr
    );
}

// Record a CopyTextureRegion for one chunk's occupancy column
// (1 × SECTIONS_PER_CHUNK × 1) into the occupancy map.
void RecordOccCopy(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* staging,
    UINT64 stagingOffset,
    ID3D12Resource* occupancy,
    int cx,
    int cz
) {
    int ax = AtlasSlot(cx);
    int az = AtlasSlot(cz);

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = staging;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = stagingOffset;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8_UINT;
    src.PlacedFootprint.Footprint.Width = 1;
    src.PlacedFootprint.Footprint.Height = SECTIONS_PER_CHUNK;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = OCC_ROW;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = occupancy;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    commandList->CopyTextureRegion(
        &dst,
        UINT(ax),
        0,
        UINT(az),
        &src,
        nullptr
    );
}