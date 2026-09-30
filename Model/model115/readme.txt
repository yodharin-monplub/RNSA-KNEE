Model115 (diagnostic, planned): per-group public-LB READOUT of model103 (public 0.947 = model76 + CoAtNet-only rank leg model75 + model100 at w 0.30), group G2 = Medial OA, Synovitis, Baker's, Fracture.
The notebook is model103's submission notebook plus one last cell that keeps only these 4 findings and sets the other 8
columns to 0.5. A constant column scores AUC 0.5 and the metric is the mean of 12 per-finding AUCs, so
    G2 mean AUC on the public set = (12 * LB - 4) / 4   (+-0.0015 from the 3-decimal rounding).
Purpose: model114-116 (model103) vs model117-119 (model76) show group by group where the CoAtNet leg helps, to set
per-group leg weights and to aim the next models at the weakest group (forum 744294: one team's CoAtNet read
ACL/MCL/menisci 0.962, Medial OA/Synovitis/Baker's/Fracture 0.941, Lateral OA/PF OA/Effusion/Contusion 0.914).
Built with Other/scripts/make_group_readout.py --source model103 --group 2 --name model115 (copy in this folder); needs a
dummy run on the main account's GPU before it can be submitted (Kaggle GPU quota refresh 2026-10-03 07:00 +07).
Score: pending.
