Model96 (built, not pushed): model78 with model75 at DOUBLE weight inside the fleet (fleet = weighted mean of member
probabilities: model72 1, model71 1, model73 1, model75 2 -> model75 = 40% of our side), leg weight unchanged at 0.30.
Why: model75 alone (model89, 0.939) beat the equal-weight fleet (model88, 0.938), and the leg only paid once model75 joined
(model80 0.944 -> model78 0.946). Gold (noise-level): fleet alone 0.9274 vs 0.9265 equal; proxy blend 0.9346 vs 0.9345.
Build: python Other/scripts/run_model80.py --name model96 --title "RSNA Knee Model96 Speedy Fleet4 Model75 x2"
--with-model75 --member-weights '{"model75": 2.0}' [--push]  (--member-weights added 2026-09-27; default builds unchanged).
Decision: pushed after tomorrow's model95 (w 0.40) result shows which direction the leg wants.
Score: not submitted.
