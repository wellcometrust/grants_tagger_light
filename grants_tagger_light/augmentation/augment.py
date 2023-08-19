import json
import multiprocessing
import os

import typer
from loguru import logger
from datasets import load_dataset
import numpy as np


from grants_tagger_light.augmentation.augment_openai import AugmentOpenAI
from grants_tagger_light.utils.years_tags_parser import parse_years

augment_app = typer.Typer()


def _count_elements_in_sublist(sublist):
    element_count = {}
    for element in sublist:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1
    return element_count


def _merge_dicts(dict_list):
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
    return merged_dict


def augment(
    data_path: str,
    save_to_path: str,
    model_key: str = 'gpt-3.5-turbo',
    num_proc: int = os.cpu_count(),
    batch_size: int = 64,
    train_years: list = None,
    test_years: list = None,
    min_examples: int = 15,
    prompt_template: str = 'grants_tagger_light/augmentation/prompt.template',
    concurrent_calls: int = 5,
    temperature: float = 1.5
):
    if model_key.strip().lower() not in ['gpt-3.5-turbo', 'text-davinci', 'gpt-4']:
        raise NotImplementedError(f"{model_key} not implemented as an augmentation framework")

    # We only have 1 file, so no sharding is available https://huggingface.co/docs/datasets/loading#multiprocessing
    dset = load_dataset("json", data_files=data_path, num_proc=1)
    # By default, any dataset loaded is set to 'train' using the previous command
    if "train" in dset:
        dset = dset["train"]

    if train_years is not None and len(train_years) > 0:
        dset = dset.filter(lambda x: any(np.isin(train_years, [str(x["year"])])), num_proc=num_proc)
    if test_years is not None and len(test_years) > 0:
        dset = dset.filter(lambda x: not any(np.isin(test_years, [str(x["year"])])), num_proc=num_proc)

    logger.info("Obtaining count values from the labels...")
    pool = multiprocessing.Pool(processes=num_proc)
    element_counts_list = pool.map(_count_elements_in_sublist, dset['meshMajor'])
    pool.close()
    pool.join()

    merged_element_counts = _merge_dicts(element_counts_list)
    sorted_merged_element_counts = sorted(merged_element_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_merged_element_counts_dict = dict(sorted_merged_element_counts)

    with open(f"{save_to_path}.count", 'w') as f:
        f.write(json.dumps(sorted_merged_element_counts_dict, indent=2))

    tags_to_augment_counts = {k: v for k, v in sorted_merged_element_counts_dict.items() if v < min_examples}
    tags_to_augment = [k for k, v in sorted_merged_element_counts_dict.items() if v < min_examples]

    biggest_tags_to_augment = [f"{k}({sorted_merged_element_counts_dict[k]})" for k in tags_to_augment[:5]]
    smallest_tags_to_augment = [f"{k}({sorted_merged_element_counts_dict[k]})" for k in tags_to_augment[-5:]]
    logger.info(f"Augmenting a total of {len(tags_to_augment)} tags, from {biggest_tags_to_augment} to "
                f"{smallest_tags_to_augment}")

    logger.info(f"Collecting existing examples of those tags to send in the prompt")
    dset = dset.filter(lambda x: any(np.isin(tags_to_augment, x["meshMajor"])), num_proc=num_proc)
    dset = dset.map(
        lambda _, y: {'idx': y},
        with_indices=True,
        batched=True,
        batch_size=batch_size,
        desc="Creating idx",
        num_proc=num_proc,
    )
    collect_concurrent_calls = []
    for t in tags_to_augment:
        if len(collect_concurrent_calls) >= concurrent_calls:
            AugmentOpenAI(prompt_template_path=prompt_template, model_key=model_key).generate(
                collect_concurrent_calls,
                dset,
                save_to_path,
                train_years,
                model_key,
                temperature=temperature,
                num_proc=num_proc,
            )
            collect_concurrent_calls = []
        else:
            if tags_to_augment_counts[t] < min_examples:
                missing = min_examples - tags_to_augment_counts[t]
                collect_concurrent_calls.append((t, missing))


@augment_app.command()
def augment_cli(
    data_path: str = typer.Argument(
        ...,
        help="Path to mesh.jsonl"),
    save_to_path: str = typer.Argument(
        ..., help="Path to save the serialized PyArrow dataset after preprocessing"
    ),
    model_key: str = typer.Option(
        "gpt-3.5-turbo",
        help="LLM to use data augmentation. By now, only `openai` is supported"
    ),
    num_proc: int = typer.Option(
        os.cpu_count(),
        help="Number of processes to use for data augmentation"
    ),
    batch_size: int = typer.Option(
        64,
        help="Preprocessing batch size (for dataset, filter, map, ...)"
    ),
    train_years: str = typer.Option(
        None,
        help="If set, Comma-separated years you want to include in the data augmentation process"
    ),
    test_years: str = typer.Option(
        None,
        help="If set, Comma-separated years you want to exclude in the data augmentation process"
    ),
    min_examples: int = typer.Option(
        15,
        help="If set, Comma-separated years you want to exclude in the data augmentation process"
    ),
    prompt_template: str = typer.Option(
        'grants_tagger_light/augmentation/prompt.template',
        help="File to use as a prompt. Make sure to ask the LLM to return a dict with two fields: `abstract` and `tags`"
    ),
    concurrent_calls: int = typer.Option(
        25,
        help="Concurrent calls with 1 tag each to the different model"
    ),
    temperature: float = typer.Option(
        1.5,
        help="A value between -2 and 2. The bigger - the more creative."
    ),
):
    if not data_path.endswith("jsonl"):
        logger.error(
            "It seems your input MeSH data is not in `jsonl` format. "
            "Please, run first `scripts/mesh_json_to_jsonlpy.`"
        )
        exit(-1)

    if float(temperature) > 2.0 or float(temperature) < -2.0:
        logger.error(
            "Temperature should be in the range [-2, 2]"
        )
        exit(-1)

    augment(data_path,
            save_to_path,
            model_key=model_key,
            num_proc=num_proc,
            batch_size=batch_size,
            train_years=parse_years(train_years),
            test_years=parse_years(test_years),
            min_examples=min_examples,
            prompt_template=prompt_template,
            concurrent_calls=concurrent_calls,
            temperature=temperature
            )
