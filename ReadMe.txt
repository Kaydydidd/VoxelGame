Desired Generation Pipeline:


initialize renderer, seed, and chunk manager -> 

generate initial player region (chunk size of 32 × 32 × 128, region size of 8 x 8 chunks) -> 

generate chunks within render distance while keeping the player region cached (As in the current 8x8 region the player is in should always be cached at full LOD in order to quickly render them, as well as the 8 surrounding regions.) -> 

build chunk data for each needed chunk -> 

per-chunk upload to GPU -> 

enter frame loop and calculate loaded chunks via player position -> 

queue new chunks on CPU -> 

upload new chunks on GPU (per chunk)