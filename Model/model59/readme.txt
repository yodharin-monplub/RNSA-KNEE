Model59: maxspan arm, 24 windows, 4-epoch cosine schedule (the last untested training dimension).

Stopped by this project's own budget guard at its 1.0 USD run cap after epoch 2 of 4 (the cap was
inherited from the 2-epoch model51 budget and is too small for four epochs). The result is
nonetheless informative: weak-validation BCE was 0.287046 after epoch 1 and 0.287058 after epoch 2,
flat, so a longer schedule does not help this fine-tune. The epoch-2 checkpoint duplicates model51's
setting and was not evaluated further or used. Estimated cost 0.79 USD; rental destroyed.
