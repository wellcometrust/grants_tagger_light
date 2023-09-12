# run in c5.9xlarge with at least 72GB of RAM
grants-tagger retag mesh data/raw/allMeSH_2021.jsonl [SET_YOUR_OUTPUT_FILE_HERE] \
  --tags "Artificial Intelligence,HIV" \
  --years 2016,2017,2018,2019,2020,2021 \
  --supervised