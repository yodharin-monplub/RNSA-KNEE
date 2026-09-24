Model19: equal probability blend of model16 and model18.
Weights: model16 0.5; model10, model12, model15 each 1/6.
Uses exact component inference scripts with sequential GPU processes.
Development AUC on 58 cases: 0.9262210133. Paired bootstrap difference
versus model12 has 95% interval [-0.0007795, 0.0138176], including zero.
These cases are upstream development data, not untouched validation.
Weights were fixed before measuring this blend; no per-label weight search.
No public leaderboard score yet. Private T4 inference v1 COMPLETE. Three visible rows passed schema,
finite probabilities, alignment, and exact blend arithmetic (1e-12 tolerance).
Runtime 113.78 seconds.
Cloud model16 training is complete and its rental destroyed.
Architecture, original checkpoints and geometry: Johnathan Wagner /
Dread Development. Fine-tuned model16: this project. See SOURCES.md.

Confirmed public AUC0.935, submission56153836, version1. New campaign best; +0.003 over model16 .932.
