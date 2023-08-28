# Augments data using a file with 1 label per line and years
grants-tagger augment mesh [FOLDER_AFTER_PREPROCESSING] [OUTPUT_FOLDER] \
  --tags-file-path [YOUR_TAGS_FILE] \
  --examples 25 \
  --concurrent-calls 25