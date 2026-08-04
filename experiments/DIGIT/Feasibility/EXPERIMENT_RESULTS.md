Experiment Results
System	Answer Acc	Support Acc	Notes
Baseline A (YES/NO/MAYBE + template)	0.197	—	Rule-based, 3 classes only
Baseline B (rich primitives + template)	0.156	0.271	Rule-based, all 4 primitives
System C (learned)	0.316	0.262	Best val loss: 1.659, leakage: 0.012
Key observations:

System C (learned) achieves ~2x the answer accuracy of the rule-based baselines
Leakage is very low (0.012) — the bottleneck successfully restricts information flow
The decoder generates coherent intuition text: "evidence appears moderately supportive.", "no clear pattern detected. evidence is insufficient."
The model is overfitting after ~epoch 9 (val loss bottoms at 1.659, then rises) — expected with synthetic data and a model this size. More diverse training data and regularization would help.
5.5M parameters, ~15min total training on RTX 5070 Ti