Model48: model43's Gemini-relabel CoAtNet views plus this project's fine-tuned DINOv2 members (model47).

Only change versus model43: find_weights points at yodharinmonplub/rsna-knee-model47-members instead of
pilkwang/rsna-knee-weights. Manifest, member ids, configs, the 20/20 fingerprint gate, the CoAt resgated
arm and the final per-finding blend are unchanged. Kaggle v1 COMPLETE (all members' fingerprints matched),
submitted 56318312.

Confirmed public AUC 0.942 - identical to model43 (arms only), confirming the member fine-tuning contributes nothing on top.
