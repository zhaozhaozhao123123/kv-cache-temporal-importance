# Synthetic experiment results

These results describe seeded synthetic Q/K/V, not Qwen or other pretrained models.
Attention is softmax(QK/sqrt(D))V. Quantization is an offline float simulation.
Temporal protection uses previous-step scores. The recent-only baseline has fewer full-precision tokens;
compare cumulative/window/EMA/random at equal token budget. EMA+outliers uses extra sparse storage,
so it is not a storage-matched comparison. 2% is a requested fraction; ceil per group can exceed it.
No measured memory compression, latency, perplexity or task accuracy is reported.
