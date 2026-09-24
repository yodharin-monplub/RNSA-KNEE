Model62: not a model - the external-data preparation step.

Streams the NYU fastMRI single-coil knee archives (train 78.0 GB, val 16.1 GB; links from the account owner's
accepted NYU Data Sharing Agreement, stored only under the ignored data/ folder) on a Vast rental, keeps the
1,173 exams annotated by fastMRI+ (microsoft/fastmri-plus, MIT / CC BY 4.0) as uint8 320x320 reconstructions,
and uploads them to the PRIVATE Kaggle dataset yodharinmonplub/rsna-knee-external-fastmri-plus-cache.
No redistribution. Source: cloud/build_fastmri_cache.py, cloud/upload_fastmri_cache.py, cloud/run_model62.sh.
Competition rule 2.6 allows free public external data; host forum reply (thread 733965) does not exclude
click-through research licences.
