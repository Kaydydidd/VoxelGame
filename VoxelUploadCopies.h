// VoxelUploadCopies.h

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>

void RecordChunkCopy(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* staging,
    UINT64 stagingOffset,
    ID3D12Resource* atlas,
    int cx,
    int cz
);

void RecordOccCopy(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* staging,
    UINT64 stagingOffset,
    ID3D12Resource* occupancy,
    int cx,
    int cz
);