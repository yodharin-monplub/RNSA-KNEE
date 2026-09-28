Model95: model78 with the fleet weight raised from 0.30 to 0.40 (one line of the leg differs). model78 = model76 +
fleet4 (model72, model71 x5, model73, model75; 0.938 alone) rank-blended at w 0.30 = 0.946 public (new best), after the
3-member fleet at 0.30 gave 0.944. Tests whether the stronger fleet deserves more weight.
Build: python Other/scripts/run_model80.py --name model95 --title "RSNA Knee Model95 Speedy Our Heads Fleet4 W40"
--with-model75 --weight 0.40 [--push].
Kaggle: yodharinmonplub/rsna-knee-model95-speedy-our-heads-fleet4-w40 v1 pushed 2026-09-27 ~22:20 (dummy run); submit
after 07:00.
Dummy run VERIFIED 2026-09-28 06:48 (from outputs): model61 Raptor weights, fleet4 leg at w 0.40 (3/3 studies, 8
checkpoints), no errors. Submit after 07:00.
SUBMITTED 2026-09-28 ~07:05 (+07), submission 56623256.
Score: public 0.945 (scored 2026-09-28 13:50) = -0.001 vs model78 (same leg at w 0.30, 0.946). A heavier fleet weight
hurts: keep w 0.30 (model80 at w 0.30 without model75 was 0.944, model83 at w 0.45 dropped by its rule).
