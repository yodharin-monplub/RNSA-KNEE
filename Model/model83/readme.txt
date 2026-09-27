Model83 (prepared, not pushed): model80 with the fleet weight raised from 0.30 to 0.45 - d4-blend's weight for its
own fleet. Everything else is model80 (one line of the fleet-leg cell differs: _M80_W = 0.45).
Built by: python Other/scripts/run_model80.py --name model83 --title "RSNA Knee Model83 Speedy Our Heads Fleet W45"
--weight 0.45 [--push].
Gold (58 cases, proxy base = model61 Raptor arms): w 0.45 +0.0015 vs w 0.30 ~+0.0013 - indistinguishable on gold.
Decision rule, fixed now: push only if model80 (w 0.30) beats model76 on the public LB, or if model81 (fleet alone)
scores >= 0.940 (a fleet that strong deserves more weight).
DROPPED 2026-09-27: model80 scored 0.944 (< model76 0.945) and model81 0.933 (< 0.940); the rule was not met.
Score: not submitted.
