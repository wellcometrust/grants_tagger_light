# Augments data using a file with 1 label per line and years
grants-tagger augment mesh [FOLDER_AFTER_PREPROCESSING] [SET_YOUR_OUTPUT_FOLDER_HERE] \
  --tags-file-path tags_to_augment.txt \
  --examples 25 \
  --concurrent-calls 25