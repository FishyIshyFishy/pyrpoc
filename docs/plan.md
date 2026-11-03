# Phased implementation plan for autonomous microscopy

## Phase 0: Fix Data structs
See `pyrpoc/core/README.md`, this amounted to a major refactor to make it clear the way the app communicates to programs, and the way programs communicate to the app. Basically, acquisition panel should not be a god object in the backend, even if it's an important user facing object. 

## Phase 1: Programs


## Phase 2: Displays

## Phase 3: Data library changes 
Enable saving + cross program stuff, like galvo calib --> survey --> loop workflow.