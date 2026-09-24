Model 5: DINOv2-small MRI fine-tuning
Status: trained on Vast.ai. Gold macro AUC: 0.767302 (58 held-out studies).
Epoch 3 selected using weak validation BCE. Local artifacts: artifacts/remote/model5/.
Starts from model2's weak-validation-selected linear head and the original
DINOv2 encoder. Unfreezes the final two transformer blocks and final norm.
Four epochs, low encoder learning rate, mild horizontal flip and intensity
augmentation. Mean/std normalization stays fixed from model2 training features.
Uses the same 3691/654/58 partition; no gold labels or gold report duplicates
enter training. The initial frozen checkpoint is a candidate, preventing a
forced switch if fine-tuning fails to improve weak validation BCE.
Run: python -m knee.finetune --config model5/config.json
This model has its own encoder weights; do not mix its head with the frozen
encoder. Sources/licenses and shared-label validation limitations: SOURCES.md.
