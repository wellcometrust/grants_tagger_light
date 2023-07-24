import json
from argparse import ArgumentParser
import numpy as np
from loguru import logger


def process_data(item, filter_tags: list = None, filter_years: list = None):
    check_tags = filter_tags is not None
    check_years = filter_years is not None

    if check_tags and 'meshMajor' not in item:
        logger.warning("`meshMajor` not found in the fields. Unable to filter tags.")
        check_tags = False

    if check_years and 'year' not in item:
        logger.warning("`year` not found in the fields. Unable to filter tags.")
        check_years = False
    if check_tags:
        if filter_tags is None:
            filter_tags = []
        if len(filter_tags) > 0 and not any(np.isin(filter_tags, item['meshMajor'])):
            return False

    if check_years:
        if filter_years is None:
            filter_years = []

        # Making sure it's str and not int
        filter_years = [str(y) for y in filter_years]
        if len(filter_years) > 0 and not any(np.isin(filter_years, [str(item['year'])])):
            return False

    return True


def mesh_json_to_jsonl(input_path, output_path, input_encoding='latin1', output_encoding='latin1',
                       filter_tags: str = '', filter_years: str = '', show_progress: bool = True):
    """
    Mesh json is not optimized for parallel processing. This script aims to transform it into `jsonl` so that,
    by having 1 json per line, you can evenly distribute it among all the cores.

    Args:
        input_path: allMeSH_2021.json (or similar) path
        output_path: path for the resulted jsonl file
        input_encoding: encoding of the input json (default `latin1`)
        output_encoding: encoding of the output jsonl (default `latin1`)
        filter_tags: tags separated by commas(T1,T2,T3) to only include the entries with  those tags
        filter_years: years separated by commas(2008,2009) to only include the entries of those years
        show_progress: print the number of line you are processing
    Returns:

    """
    filter_tags_list = list(filter(lambda x: x.strip() != '', filter_tags.split(',')))
    filter_years_list = list(filter(lambda x: x.strip() != '', filter_years.split(',')))
    with open(output_path, 'w', encoding=output_encoding) as fw:
        with open(input_path, 'r', encoding=input_encoding) as fr:
            for idx, line in enumerate(fr):
                if show_progress:
                    print(idx, end='\r')
                # Skip 1st line
                if idx == 0:
                    logger.info(f"Skipping first line (articles): {line}")
                    continue
                try:
                    sample = json.loads(line[:-2])
                except:
                    logger.warning(f"Skipping line in bad json format: {line}")
                    continue
                if process_data(sample, filter_tags_list, filter_years_list):
                    fw.write(json.dumps(sample))
                    fw.write('\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", required=True, help="path to input allMeSH_2021.json (or equivalent)")
    parser.add_argument("--output_path", required=True, help="path to ioutput jsonl")
    parser.add_argument("--input_encoding", required=False, default='latin1', help="encoding of the input json")
    parser.add_argument("--output_encoding", required=False, default='latin1', help="encoding of the output jsonl")
    parser.add_argument("--filter_tags", required=False, default='',
                        help="comma-separated tags to include (the rest will be discarded)")
    parser.add_argument("--filter_years", required=False, default='',
                        help="comma-separated years to include (the rest will be discarded)")

    args = parser.parse_args()

    mesh_json_to_jsonl(args.input_path, args.output_path, args.input_encoding, args.output_encoding, args.filter_tags,
                       args.filter_years)
