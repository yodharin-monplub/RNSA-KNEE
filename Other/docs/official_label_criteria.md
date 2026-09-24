# Official label criteria (host overview thread, discussion 733343, read 2026-09-19)

Source: "Knee Abnormality Detection AI Challenge Overview", Po-Hao "Howard" Chen (host), courtesy Dr. Jacob Kazam.
Summarised, not verbatim.

General: twelve binary exam-level labels for a single knee. Two subspecialty MSK radiologists labelled each
reference study independently from the images; a third adjudicated. Borderline / "on the fence" findings were
graded NEGATIVE (favouring specificity).

- ACL tear: high-grade partial (>50% of fibres) or full-thickness tear. Mild signal change, degeneration or
  thickening without discontinuity = negative.
- MCL tear: high-grade partial or complete ACUTE tear with disrupted fibres and edema. Low-grade sprains and
  chronic/remote changes = negative.
- Medial / Lateral meniscus tear: signal definitely contacting the surface on >= 2 images, or morphologic
  abnormality (truncated, diminutive, displaced fragment). Intrasubstance degeneration = negative.
- Medial / Lateral / Patellofemoral OA: moderate or large area (about 1 cm or more) of HIGH-GRADE cartilage loss
  (>50% thickness), with or without subchondral marrow change.
- Joint effusion: MODERATE or LARGE amount of fluid distending the joint.
- Synovitis: inflammation and thickening of the synovial lining.
- Baker cyst: MODERATE or LARGE fluid collection in the characteristic location.
- Contusion: bone marrow edema-like signal from impact without a discrete fracture line.
- Acute fracture: ACUTE cortical break or fracture line.

## Where scripts/llm_relabel.py's prompt contradicts this
- MCL: prompt counts "sprain grade 1-3"; official counts only high-grade acute tears.
- Baker's: prompt says "any size"; official says moderate or large.
- ACL: prompt includes partial tears generally; official requires high-grade (>50%) or complete.
- OA: prompt says "meaningful grade, osteophytes"; official requires >50% thickness loss over ~1 cm+.
- Fracture: prompt includes subchondral insufficiency/any fracture; official says acute.
- Missing rule: borderline findings are negative.

## Other forum/rules facts
- Rules 2.6: external data allowed if publicly available and equally accessible at no cost to all participants
  (or "reasonable" minimal cost). Winner licence CC-BY-NC 4.0; incompatible-licence data/models need not be open-sourced.
- Pinned host thread 733965: hosted commercial LLMs/APIs are explicitly permitted for processing competition data
  (e.g. extracting labels from reports); not considered private sharing.
- Forum (735826): 4th-place competitor says even regex labels go far; others note gold effusion labels are noisy
  around "small" (both 0 and 1 for the same template sentence); hosts' readers did not see the reports.
