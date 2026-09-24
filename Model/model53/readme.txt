Model53: model43 with a higher base weight for the CoAtNet side of the final per-finding blend.

Change: _coatnet_weight base 0.60 -> 0.70 for the seven findings that use the base value; the published
exceptions (ACL 0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00, Lateral OA 0.75, Fracture 0.75) are
unchanged, as are all seven Gemini-relabel CoAtNet views and the transformer side.

Rationale fixed in advance: our CoAtNet views are trained on labels that agree better with the expert
gold labels (0.9035 vs 0.8927), so the side that improved should arguably carry more weight. This is a
single pre-declared step, not a weight search.

Kaggle v1 COMPLETE, submitted 56327619.

Confirmed public AUC 0.942 - identical to model43, so raising the base CoAtNet weight from 0.60 to 0.70 changes nothing measurable. The published weights are kept.
