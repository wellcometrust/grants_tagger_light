import json
import multiprocessing
import os

import typer
from loguru import logger
import numpy as np


from grants_tagger_light.augmentation.augment_openai import AugmentOpenAI

from datasets import load_from_disk

from grants_tagger_light.augmentation.parallel_augment_openai import (
    ParallelAugmentOpenAI,
)
from grants_tagger_light.utils.years_tags_parser import parse_tags

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
    model_key: str = "gpt-3.5-turbo",
    num_proc: int = os.cpu_count(),
    batch_size: int = 64,
    min_examples: int = None,
    examples: int = 25,
    prompt_template: str = "grants_tagger_light/augmentation/prompt.template",
    concurrent_calls: int = os.cpu_count() * 2,
    temperature: float = 1.5,
    tags: list = None,
    tags_file_path: str = None,
):
    if model_key.strip().lower() not in ["gpt-3.5-turbo", "text-davinci", "gpt-4"]:
        raise NotImplementedError(
            f"{model_key} not implemented as an augmentation framework"
        )

    dset = load_from_disk(os.path.join(data_path, "dataset"))
    if "train" in dset:
        dset = dset["train"]

    logger.info("Obtaining count values from the labels...")
    pool = multiprocessing.Pool(processes=num_proc)
    element_counts_list = pool.map(_count_elements_in_sublist, dset["meshMajor"])
    pool.close()
    pool.join()

    merged_element_counts = _merge_dicts(element_counts_list)
    sorted_merged_element_counts = sorted(
        merged_element_counts.items(), key=lambda x: x[1], reverse=True
    )
    sorted_merged_element_counts_dict = dict(sorted_merged_element_counts)

    print(f"Tags: {tags}")
    if tags is None:
        tags = []
    if tags_file_path is not None:
        with open(tags_file_path, "r") as f:
            tags.extend([x.strip() for x in f.readlines()])
            logger.info(
                f"Tags file path found. Filtering {len(tags)} tags "
                f"(examples found: {tags[:15]}...)"
            )
    if len(tags) > 0:
        sorted_merged_element_counts_dict = {
            k: v for k, v in sorted_merged_element_counts_dict.items() if k in tags
        }
        logger.info(f"Tags count dictionary: {sorted_merged_element_counts_dict}")

    if min_examples is not None:
        sorted_merged_element_counts_dict = {
            k: v
            for k, v in sorted_merged_element_counts_dict.items()
            if v < min_examples
        }

    if len(sorted_merged_element_counts_dict.keys()) < 1:
        logger.error(
            "I did not find any examples for your tags "
            "in your preprocessed folder. Try:\n"
            "- Other train/set split in `preprocess`;\n"
            "- Other years;\n"
            "- Other tags;"
        )
        exit(-1)

    with open(f"{save_to_path}.count", "w") as f:
        f.write(json.dumps(sorted_merged_element_counts_dict, indent=2))

    tags_to_augment = list(sorted_merged_element_counts_dict.keys())

    if len(tags_to_augment) < concurrent_calls:
        logger.error(
            "Found less tags than concurrent calls to OpenAI."
            f" Overwritting `concurrent-calls` to {len(tags_to_augment)}"
        )
        concurrent_calls = len(tags_to_augment)

    biggest_tags_to_augment = [
        f"{k}({sorted_merged_element_counts_dict[k]})" for k in tags_to_augment[:5]
    ]
    smallest_tags_to_augment = [
        f"{k}({sorted_merged_element_counts_dict[k]})" for k in tags_to_augment[-5:]
    ]

    logger.info(
        f"Augmenting a total of {len(tags_to_augment)} tags, "
        f"from {biggest_tags_to_augment} to {smallest_tags_to_augment}"
    )

    logger.info("Collecting existing examples of those tags to send in the prompt")

    dset = dset.filter(
        lambda x: any(np.isin(tags_to_augment, x["meshMajor"])), num_proc=num_proc
    )

    dset = dset.map(
        lambda _, y: {"idx": y},
        with_indices=True,
        batched=True,
        batch_size=batch_size,
        desc="Creating idx",
        num_proc=num_proc,
    )

    if concurrent_calls == 1:
        openai = AugmentOpenAI
    else:
        openai = ParallelAugmentOpenAI

    collect_concurrent_calls = []

    for t in tags_to_augment:
        if len(collect_concurrent_calls) >= concurrent_calls:
            openai(prompt_template_path=prompt_template, model_key=model_key).generate(
                collect_concurrent_calls,
                dset,
                save_to_path,
                model_key,
                temperature=temperature,
                num_proc=num_proc,
            )
            collect_concurrent_calls = []
        else:
            collect_concurrent_calls.append((t, examples))

    # Remaining rows of the last batch
    if len(collect_concurrent_calls) > 0:
        openai(prompt_template_path=prompt_template, model_key=model_key).generate(
            collect_concurrent_calls,
            dset,
            save_to_path,
            model_key,
            temperature=temperature,
            num_proc=num_proc,
        )


@augment_app.command()
def augment_cli(
    data_path: str = typer.Argument(..., help="Path to mesh.jsonl"),
    save_to_path: str = typer.Argument(..., help="Path to save the new jsonl data"),
    model_key: str = typer.Option(
        "gpt-3.5-turbo",
        help="LLM to use data augmentation. By now, only `openai` is supported",
    ),
    num_proc: int = typer.Option(
        os.cpu_count(), help="Number of processes to use for data augmentation"
    ),
    batch_size: int = typer.Option(
        64, help="Preprocessing batch size (for dataset, filter, map, ...)"
    ),
    min_examples: int = typer.Option(
        None,
        help="Minimum number of examples to require. "
        "Less than that will trigger data augmentation.",
    ),
    examples: int = typer.Option(25, help="Examples to generate per each tag."),
    prompt_template: str = typer.Option(
        "grants_tagger_light/augmentation/prompt.template",
        help="File to use as a prompt. "
        "Make sure to ask the LLM to return a dict with two fields: "
        "`abstract` and `tags`",
    ),
    concurrent_calls: int = typer.Option(
        os.cpu_count() * 2,
        min=1,
        help="Concurrent calls with 1 tag each to the different model",
    ),
    temperature: float = typer.Option(
        1.5,
        min=0,
        max=2,
        help="A value between 0 and 2. The bigger - the more creative.",
    ),
    tags: str = typer.Option(None, help="Comma separated list of tags to retag"),
    tags_file_path: str = typer.Option(
        None,
        help="Text file containing one line per tag to be considered. "
        "The rest will be discarded.",
    ),
):
    if not os.path.isdir(data_path):
        logger.error(
            "The data path should be a folder with saved data from "
            "`preprocessing` step."
        )
        exit(-1)

    if tags_file_path is None and tags is None and min_examples is None:
        logger.error(
            "To understand which tags need to be augmented, "
            "set either --min-examples or --tags-file-path or --tags"
        )
        exit(-1)

    if tags_file_path is not None and not os.path.isfile(tags_file_path):
        logger.error(f"{tags_file_path} not found")
        exit(-1)

    if float(temperature) > 2.0 or float(temperature) < -2.0:
        logger.error("Temperature should be in the range [-2, 2]")
        exit(-1)

    augment(
        data_path,
        save_to_path,
        model_key=model_key,
        num_proc=num_proc,
        batch_size=batch_size,
        min_examples=min_examples,
        examples=examples,
        prompt_template=prompt_template,
        concurrent_calls=concurrent_calls,
        temperature=temperature,
        tags=parse_tags(tags),
        tags_file_path=tags_file_path,
    )
