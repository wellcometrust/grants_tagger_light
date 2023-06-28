import json
import argparse
from tqdm import tqdm


def convert_json_to_jsonl(json_path, jsonl_path):
    # First count the lines
    with open(json_path, "r", encoding="latin1") as input_file:
        num_lines = sum(1 for line in input_file)

    # Skip 1st line
    with open(json_path, "r", encoding="latin1") as input_file, open(
        jsonl_path, "w"
    ) as output_file:
        pbar = tqdm(total=num_lines)
        for idx, line in enumerate(input_file):
            if idx == 0:
                continue
            sample = json.loads(line[:-2])
            output_file.write(json.dumps(sample) + "\n")
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default="data/raw/allMeSH_2021.json")
    parser.add_argument("--jsonl-path", type=str, default="data/raw/allMeSH_2021.jsonl")
    args = parser.parse_args()

    convert_json_to_jsonl(args.json_path, args.jsonl_path)
