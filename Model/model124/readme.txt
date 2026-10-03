Model124: model121 (model76 + six-member CoAtNet-2 384 leg v4: model110f0/2, model104/2, model75/1, model109/2,
model100/2, model98/2) with the final rank-leg weight w 0.50 instead of 0.30.
Why: at w 0.30 two, three and six strong members all scored 0.947 (model103/107/121) and model104 alone scored 0.942,
close to Speedy (0.943): our side may deserve a larger share. Weight curve: model123 (0.40), model124 (0.50), model125 (0.60).
Build: as model121 with --name model124 --weight 0.50 --title "RSNA Knee Model124 Leg Six Stride2 W500"
(run_speedy_fleet_v2.py --leg v4).
Score: pending.
