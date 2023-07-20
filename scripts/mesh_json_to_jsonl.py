import json
from argparse import ArgumentParser


def mesh_to_jsonl(input_path, output_path, input_encoding='latin1', output_encoding='latin1'):
    """
    Mesh json is not optimized for parallel processing. This script aims to transform it into `jsonl` so that,
    by having 1 json per line, you can evenly distribute it among all the cores.

    Args:
        input_path: allMeSH_2021.json (or similar) path
        output_path: path for the resulted jsonl file
        input_encoding: encoding of the input json (default `latin1`)
        output_encoding: encoding of the output jsonl (default `latin1`)

    Returns:

    """
    with open(output_path, 'w', encoding=output_encoding) as fw:
        with open(input_path, 'r', encoding=input_encoding) as fr:
            for idx, line in enumerate(fr):
                print(idx, end='\r')
                # Skip 1st line
                if idx == 0:
                    continue
                sample = json.loads(line[:-2])

                fw.write(json.dumps(sample))
                fw.write('\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", required=True, help="path to input allMeSH_2021.json (or equivalent)")
    parser.add_argument("--output_path", required=True, help="path to input jsonl")
    parser.add_argument("--input_encoding", required=False, default='latin1', help="encoding of the input json")
    parser.add_argument("--output_encoding", required=False, default='latin1', help="encoding of the output jsonl")

    args = parser.parse_args()

    mesh_to_jsonl(args.input_path, args.output_path, args.input_encoding, args.output_encoding)
