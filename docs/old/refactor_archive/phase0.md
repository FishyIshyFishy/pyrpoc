### Phase 0 — Record what the current code does, and rollback features to be eliminated for v3.1

Tasks:
1. Survey the codebase and develop a comprehensive understanding of everything that is implemented. If there are questions about what things look like currently, on the user experience side, ask. Make no assumptions. 
2. Eliminate the simulated fallbacks. If acquisition fails, it should fail hard. 
3. Eliminate everything related to automatic segmentation, as well as anything that causes the cellpose/torch dependency. Then remove that dependency from the uv project as well. 
4. Eliminate everything related to streaming (streamed_image_display.py and things that coordinate with it, and flim_display.py). Per the 260827-refactor_plan.md, the streaming is something we are architecturally keeping an option for the future, but not something we want to consider in this migration.
5. Get rid of the full @pyrpoc/rpoc/ folder. It contains old, deprecated code, which we don't care to have. The actual optocontrol editor widget is already created in @pyrpoc/optocontrols/mask.py . 
6. If anything else looks strange as you survey, ask about this for clarification on what is going on. 

