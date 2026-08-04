# Stage 15C Variant Summary

variant | routing | filter | mask_pretrain | support_warmup | N | F1 | clone_cosine | joint_beats_single | answer_window_hit | support_window_hit
--- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: 
scratch_distinct_multi_support | learned_distinct | multi_support | False | False | 4 | 0.1536 | 0.668218 | True | 1.0000 | 1.0000
curriculum_pretrain_distinct_multi_support | learned_distinct | multi_support | True | False | 4 | 0.1604 | 0.838012 | True | 1.0000 | 1.0000
curriculum_pretrain_support_distinct_multi_support | learned_distinct | multi_support | True | True | 4 | 0.1604 | 0.842954 | True | 1.0000 | 1.0000
