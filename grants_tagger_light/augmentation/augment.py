import json
import multiprocessing
import os

import typer
from loguru import logger
import numpy as np


from grants_tagger_light.augmentation.augment_openai import AugmentOpenAI
from grants_tagger_light.utils.years_tags_parser import parse_years

from datasets import load_from_disk

augment_app = typer.Typer()


def _map_id_to_labels(ids, id2label):
    return [id2label[i] for i in ids]


def _restore_meshmajor(sample, id2label):
    return {"meshMajor": [_map_id_to_labels(x, id2label) for x in sample["label_ids"]]}


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
    min_examples: int = None,
    examples: int = 25,
    prompt_template: str = 'grants_tagger_light/augmentation/prompt.template',
    concurrent_calls: int = os.cpu_count()*2,
    temperature: float = 1.5,
    tags_file_path: str = None,
):
    if model_key.strip().lower() not in ['gpt-3.5-turbo', 'text-davinci', 'gpt-4']:
        raise NotImplementedError(f"{model_key} not implemented as an augmentation framework")

    dset = load_from_disk(os.path.join(data_path, "dataset"))
    if "train" in dset:
        dset = dset["train"]

    with open(os.path.join(data_path, "label2id"), "r") as f:
        label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}

    dset = dset.map(
        _restore_meshmajor,
        batched=True,
        batch_size=batch_size,
        desc="Decoding labels",
        num_proc=num_proc,
        fn_kwargs={"id2label": id2label}
    )

    logger.info("Obtaining count values from the labels...")
    pool = multiprocessing.Pool(processes=num_proc)
    element_counts_list = pool.map(_count_elements_in_sublist, dset['meshMajor'])
    pool.close()
    pool.join()

    merged_element_counts = _merge_dicts(element_counts_list)
    sorted_merged_element_counts = sorted(merged_element_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_merged_element_counts_dict = dict(sorted_merged_element_counts)
    if tags_file_path is not None:
        with open(tags_file_path, 'r') as f:
            tags = f.read().split('\n')
            logger.info(f"Tags file path found. Filtering tags (examples found: {tags[:15]}...)")
            sorted_merged_element_counts_dict = {k: v for k, v in sorted_merged_element_counts_dict.items()
                                                 if k in tags}

    if min_examples is not None:
        sorted_merged_element_counts_dict = {k: v for k, v in sorted_merged_element_counts_dict.items()
                                             if v < min_examples}

    with open(f"{save_to_path}.count", 'w') as f:
        f.write(json.dumps(sorted_merged_element_counts_dict, indent=2))

    tags_to_augment = sorted_merged_element_counts_dict.keys()

    biggest_tags_to_augment = [f"{k}({sorted_merged_element_counts_dict[k]})"
                               for k in tags_to_augment[:5]]
    smallest_tags_to_augment = [f"{k}({sorted_merged_element_counts_dict[k]})"
                                for k in tags_to_augment[-5:]]

    logger.info(f"Augmenting a total of {len(tags_to_augment)} tags, "
                f"from {biggest_tags_to_augment} to {smallest_tags_to_augment}")

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
                model_key,
                temperature=temperature,
                num_proc=num_proc,
            )
            collect_concurrent_calls = []
        else:
            collect_concurrent_calls.append((t, examples))

    if len(collect_concurrent_calls) > 0:
        AugmentOpenAI(prompt_template_path=prompt_template, model_key=model_key).generate(
            collect_concurrent_calls,
            dset,
            save_to_path,
            model_key,
            temperature=temperature,
            num_proc=num_proc,
        )


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
    min_examples: int = typer.Option(
        None,
        help="Minimum number of examples to require. Less than that will trigger data augmentation."
    ),
    examples: int = typer.Option(
        25,
        help="Examples to generate per each tag."
    ),
    prompt_template: str = typer.Option(
        'grants_tagger_light/augmentation/prompt.template',
        help="File to use as a prompt. Make sure to ask the LLM to return a dict with two fields: `abstract` and `tags`"
    ),
    concurrent_calls: int = typer.Option(
        os.cpu_count()*2,
        help="Concurrent calls with 1 tag each to the different model"
    ),
    temperature: float = typer.Option(
        1.5,
        help="A value between -2 and 2. The bigger - the more creative."
    ),
    tags_file_path: str = typer.Option(
        None,
        help="Text file containing one line per tag to be considered. The rest will be discarded."
    )
):
    if not os.path.isdir(data_path):
        logger.error(
            "The data path should be a folder with saved data from `preprocessing` step."
        )
        exit(-1)

    if tags_file_path is None and min_examples is None:
        logger.error(
            "To understand which tags need to be augmented, set either --min-examples or --tags-file-path"
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
            min_examples=min_examples,
            examples=examples,
            prompt_template=prompt_template,
            concurrent_calls=concurrent_calls,
            temperature=temperature,
            tags_file_path=tags_file_path
            )
