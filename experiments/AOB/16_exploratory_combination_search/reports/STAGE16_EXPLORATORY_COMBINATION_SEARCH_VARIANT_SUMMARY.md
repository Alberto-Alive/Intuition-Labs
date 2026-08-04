# Stage 16 Variant Summary

variant | routing | comm | filter | mask_pretrain | support_warmup | N | F1 | clone_cosine | joint_beats_single | answer_window_hit | support_window_hit
--- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: 
scratch_continuous_distinct_multi_support | learned_distinct | continuous | multi_support | False | False | 4 | 0.1748 | 0.681778 | True | 1.0000 | 1.0000
curriculum_continuous_distinct_multi_support | learned_distinct | continuous | multi_support | True | False | 4 | 0.1237 | 0.886934 | True | 1.0000 | 1.0000
curriculum_discrete_distinct_multi_support | learned_distinct | discrete_message | multi_support | True | False | 4 | 0.1237 | 0.886934 | True | 1.0000 | 1.0000
curriculum_continuous_oracle_support_all | oracle_support | continuous | all | True | False | 4 | 0.1065 | 0.950755 | False | 0.9344 | 1.0000
curriculum_discrete_oracle_support_all | oracle_support | discrete_message | all | True | False | 4 | 0.1065 | 0.950755 | False | 0.9344 | 1.0000
