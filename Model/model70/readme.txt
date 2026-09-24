Model70: model61 (public 0.943) with ONE change - the fine-spacing CoAtNet arm (finespacing-ft22, head-retrained
checkpoint model61_fine_best.pt, blend weight 0.15 of 1.30) runs at the slice geometry its author documents for the
v9 fine-spacing corpus: 80 slices per study over 2-98% of each series (slots sag-fluid 22 / sag 18 / cor-fluid 15 /
cor 10 / axial 15), 78 three-slice windows. model61 ran it at 64 slices over 6-94% with 62 windows, because the
dataset page omitted the geometry when model10 was built (Other/EXPERIMENTS.md, "GEOMETRY BUG ON OUR SIDE").

Evidence (58 gold cases, never trained on; Other/scripts/run_geometry_gold.py fine + compare_geometry_gold.py,
Kaggle CPU kernel rsna-knee-geometry-gold-fine, 2026-09-22):
  fine arm            0.9181 (current geometry, reproduces the saved value exactly) -> 0.9252 (+0.0071)
  four-arm blend      0.9326 -> 0.9340 (+0.0015, paired bootstrap 95% CI [-0.0007, +0.0042], P>0 0.91)
  biggest per-finding gains: MCL +0.057, Lateral Meniscus +0.029, Fracture +0.025
Controls: widening only the span (64 slices, 2-98%) -0.0019; 80 slices with the extra ones on the axial plane -0.0030.
The same 80-slice change on the maxspan arm gave only +0.0003 (that arm's weights were fitted at 64 slices), so only
the fine-spacing arm changes. Adopted under the rule declared before the test (blend >= +0.002, or arm >= +0.002 with
the blend not worse).

Build: python Other/scripts/run_model23_24.py model70  (kind = model61 + FINE80 patch). The fine-spacing arm now has
its own decode group (80 slices), adding roughly a quarter of that arm's decode/inference time; model61 scored in
~5.4 h of the 9 h limit. The build also applies COAT_ROOT_NEW so the public CoAtNet child finds the competition data
under either mount layout (needed on the teammate account).
Inputs: identical to model61 (17 datasets, 4 head-retrain kernels). Pushed from teammate account krittanutsomtuas
(same team) because yodharinmonplub's weekly Kaggle GPU quota is used up. Source: notebook.ipynb here.
