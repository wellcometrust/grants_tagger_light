import logging
import random

import typer
from loguru import logger

from datasets import load_dataset


from johnsnowlabs import nlp

import os

from sklearn.metrics import classification_report
from pyspark.sql.functions import *

spark = nlp.start()

retag_app = typer.Typer()


def _load_data(dset: list[str], limit=100, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    random.Random(42).shuffle(dset)
    train_size = int(split * limit)
    train_dset = dset[:train_size]
    test_dset = dset[train_size:limit]
    return train_dset, test_dset


def retag(
    data_path: str,
    save_to_path: str,
    model_key: str = "gpt-3.5-turbo",
    num_proc: int = os.cpu_count(),
    batch_size: int = 64,
    concurrent_calls: int = os.cpu_count() * 2,
    tags_file_path: str = None,
):
    if model_key.strip().lower() not in ["gpt-3.5-turbo", "text-davinci", "gpt-4"]:
        raise NotImplementedError(
            f"{model_key} not implemented as an augmentation framework"
        )

    # We only have 1 file, so no sharding is available https://huggingface.co/docs/datasets/loading#multiprocessing
    logging.info("Loading the MeSH jsonl...")
    dset = load_dataset("json", data_files=data_path, num_proc=1)
    if "train" in dset:
        dset = dset["train"]

    with open(tags_file_path, 'r') as f:
        tags = [x.strip() for x in f.readlines()]

    for tag in tags:
        logging.info(f"Retagging: {tag}")

        logging.info(f"Obtaining positive examples for {tag}...")
        positive_dset = dset.filter(
            lambda x: tag in x["meshMajor"], num_proc=num_proc
        )
        pos_x_train, pos_x_test = _load_data(positive_dset['abstractText'], limit=100, split=0.8)

        logging.info(f"Obtaining negative examples for {tag}...")
        negative_dset = dset.filter(
            lambda x: tag not in x["meshMajor"], num_proc=num_proc
        )
        neg_x_train, neg_x_test = _load_data(negative_dset['abstractText'], limit=100, split=0.8)

        train_data = [(x, tag) for x in pos_x_train]
        train_data.extend([(x, 'other') for x in neg_x_train])

        columns = ["text", "category"]
        train_df = spark.createDataFrame(train_data, columns)

        test_data = [(x, tag) for x in pos_x_test]
        test_data.extend([(x, 'other') for x in neg_x_test])
        test_df = spark.createDataFrame(test_data, columns)

        logging.info(train_df.groupBy("category") \
            .count() \
            .orderBy(col("count").desc()) \
            .show())

        logging.info(train_df.groupBy("category") \
              .count() \
              .orderBy(col("count").desc()) \
              .show())

        document_assembler = nlp.DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
            .setInputCols(["document"]) \
            .setOutputCol("sentence_embeddings") \

        classsifierdl = nlp.ClassifierDLApproach() \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("label") \
            .setLabelColumn("category") \
            .setMaxEpochs(10) \
            .setLr(0.001) \
            .setBatchSize(1) \
            .setEnableOutputLogs(True)
        # .setOutputLogsPath('logs')

        bert_clf_pipeline = nlp.Pipeline(stages=[document_assembler,
                                             embeddings,
                                             classsifierdl])

        clf_pipelineModel = bert_clf_pipeline.fit(train_df)
        preds = clf_pipelineModel.transform(test_df)
        logging.info(preds.select('category', 'text', 'label.result').show(10, truncate=80))

        preds_df = preds.select('category', 'text', 'label.result').toPandas()

        # The result is an array since in Spark NLP you can have multiple sentences.
        # Let's explode the array and get the item(s) inside of result column out
        preds_df['result'] = preds_df['result'].apply(lambda x: x[0])

        print(classification_report(preds_df['category'], preds_df['result']))



@retag_app.command()
def retag_cli(
    data_path: str = typer.Argument(..., help="Path to mesh.jsonl"),
    save_to_path: str = typer.Argument(
        ..., help="Path where to save the retagged data"
    ),
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
    concurrent_calls: int = typer.Option(
        os.cpu_count() * 2,
        min=1,
        help="Concurrent calls with 1 tag each to the different model",
    ),
    tags_file_path: str = typer.Option(
        None,
        help="Text file containing one line per tag to be considered. "
        "The rest will be discarded.",
    ),
):
    if not data_path.endswith("jsonl"):
        logger.error(
            "It seems your input MeSH data is not in `jsonl` format. "
            "Please, run first `scripts/mesh_json_to_jsonl.py.`"
        )
        exit(-1)

    if tags_file_path is None:
        logger.error(
            "To understand which tags need to be augmented set the path to the tags file in --tags-file-path"
        )
        exit(-1)

    retag(
        data_path,
        save_to_path,
        model_key=model_key,
        num_proc=num_proc,
        batch_size=batch_size,
        concurrent_calls=concurrent_calls,
        tags_file_path=tags_file_path,
    )
