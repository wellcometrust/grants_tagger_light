import json
import logging
import random

import typer
from loguru import logger

from datasets import load_dataset

from johnsnowlabs import nlp

import os

from sklearn.metrics import classification_report
from pyspark.sql.functions import col

spark = nlp.start()

retag_app = typer.Typer()


def _load_data(dset: list[str], limit=100, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    limit = min(len(dset), limit)
    random.Random(42).shuffle(dset)
    train_size = int(split * limit)
    train_dset = dset[:train_size]
    test_dset = dset[train_size:limit]
    return train_dset, test_dset


def _process_prediction_batch(save_to_path, current_batch, lightpipeline, threshold, tag, dset):
    with open(f"{save_to_path}.{tag}.jsonl", "a") as f:
        batch_texts = [x[0] for x in current_batch]
        batch_tags = [x[1] for x in current_batch]
        batch_row_nums = [x[2] for x in current_batch]
        result = lightpipeline.fullAnnotate(batch_texts)
        for r in range(len(result)):
            prediction = result[r]['label'][0].result
            prediction_confidence = float(result[r]['label'][0].metadata[tag])
            prediction_old_tags = batch_tags[r]

            if prediction_confidence < threshold:
                continue

            before = tag in prediction_old_tags
            after = prediction == tag
            dset_row = dset[batch_row_nums[r]]
            if before != after:
                if 'correction' not in dset_row:
                    dset_row['correction'] = []
                if after is True:
                    dset_row['meshMajor'].append(tag)
                    dset_row['correction'].append({'change': f"+{tag}", 'confidence': prediction_confidence})
                else:
                    dset_row['meshMajor'].remove(tag)
                    dset_row['correction'].append({'change': f"-{tag}", 'confidence': prediction_confidence})
                # logging.info(f"- Corrected: {dset_row['correction']}")
                json.dump(dset_row, f)
                f.write("\n")
        f.flush()


def retag(
    data_path: str,
    save_to_path: str,
    num_proc: int = os.cpu_count(),
    batch_size: int = 64,
    tags_file_path: str = None,
    threshold: float = 0.8
):
    # We only have 1 file, so no sharding is available https://huggingface.co/docs/datasets/loading#multiprocessing
    logging.info("Loading the MeSH jsonl...")
    dset = load_dataset("json", data_files=data_path, num_proc=1)
    if "train" in dset:
        dset = dset["train"]

    with open(tags_file_path, 'r') as f:
        tags = [x.strip() for x in f.readlines()]

    for tag in tags:
        logging.info(f"Retagging: {tag}")

        logging.info(f"- Obtaining positive examples for {tag}...")
        positive_dset = dset.filter(
            lambda x: tag in x["meshMajor"], num_proc=num_proc
        )

        if len(positive_dset['abstractText']) < 50:
            logging.info(f"Skipping {tag}: low examples ({len(positive_dset['abstractText'])}. "
                         f"Check {save_to_path}.err for more information about skipped tags.")
            with open(f"{save_to_path}.err", 'a') as f:
                f.write(tag)
            continue

        pos_x_train, pos_x_test = _load_data(positive_dset['abstractText'], limit=250, split=0.8)

        logging.info(f"- Obtaining negative examples for {tag}...")
        negative_dset = dset.filter(
            lambda x: tag not in x["meshMajor"], num_proc=num_proc
        )
        neg_x_train, neg_x_test = _load_data(negative_dset['abstractText'], limit=250, split=0.8)

        train_data = [(x, tag) for x in pos_x_train]
        train_data.extend([(x, 'other') for x in neg_x_train])

        columns = ["text", "category"]
        train_df = spark.createDataFrame(train_data, columns)

        test_data = [(x, tag) for x in pos_x_test]
        test_data.extend([(x, 'other') for x in neg_x_test])
        test_df = spark.createDataFrame(test_data, columns)

        train_df.groupBy("category")\
            .count()\
            .orderBy(col("count").desc())

        train_df.groupBy("category")\
            .count()\
            .orderBy(col("count").desc())

        document_assembler = nlp.DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
            .setInputCols(["document"]) \
            .setOutputCol("sentence_embeddings") \

        # I'm limiting the batch size to 8 since there are not many examples and big batch sizes will decrease accuracy
        classifierdl = nlp.ClassifierDLApproach() \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("label") \
            .setLabelColumn("category") \
            .setMaxEpochs(25) \
            .setLr(0.001) \
            .setBatchSize(max(batch_size, 8)) \
            .setEnableOutputLogs(True)
        # .setOutputLogsPath('logs')

        clf_pipeline = nlp.Pipeline(stages=[document_assembler,
                                            embeddings,
                                            classifierdl])

        fit_clf_pipeline = clf_pipeline.fit(train_df)
        preds = fit_clf_pipeline.transform(test_df)
        logging.info(preds.select('category', 'text', 'label.result').show(10, truncate=80))
        preds_df = preds.select('category', 'text', 'label.result').toPandas()
        preds_df['result'] = preds_df['result'].apply(lambda x: x[0])
        logging.info(classification_report(preds_df['category'], preds_df['result']))

        logging.info("- Loading the model for prediction...")
        fit_clf_pipeline.stages[-1].write().overwrite().save('clf_tmp')
        fit_clf_model = nlp.ClassifierDLModel.load('clf_tmp')

        pred_pipeline = nlp.Pipeline(stages=[document_assembler,
                                             embeddings,
                                             fit_clf_model])
        pred_df = spark.createDataFrame([['']]).toDF("text")
        fit_pred_pipeline = pred_pipeline.fit(pred_df)
        fit_pred_lightpipeline = nlp.LightPipeline(fit_pred_pipeline)
        logging.info(f"- Retagging {tag}...")

        row_counter = -1
        batch_counter = -1

        batch_total = len(dset["abstractText"])

        current_batch = []

        for text, old_tags in zip(dset["abstractText"], dset["meshMajor"]):
            row_counter += 1
            if len(current_batch) < batch_size:
                current_batch.append((text, old_tags, row_counter))
                continue
            else:
                batch_counter += 1
                print(f"Processing batch {batch_counter}/{batch_total}", end="\r", flush=True)

                _process_prediction_batch(save_to_path, current_batch, fit_pred_lightpipeline, threshold, tag, dset)
                current_batch = []

        # Remaining
        if len(current_batch) > 0:
            _process_prediction_batch(save_to_path, current_batch, fit_pred_lightpipeline, threshold, tag, dset)


@retag_app.command()
def retag_cli(
    data_path: str = typer.Argument(..., help="Path to mesh.jsonl"),
    save_to_path: str = typer.Argument(
        ..., help="Path where to save the retagged data"
    ),
    num_proc: int = typer.Option(
        os.cpu_count(), help="Number of processes to use for data augmentation"
    ),
    batch_size: int = typer.Option(
        64, help="Preprocessing batch size (for dataset, filter, map, ...)"
    ),
    tags_file_path: str = typer.Option(
        None,
        help="Text file containing one line per tag to be considered. "
        "The rest will be discarded.",
    ),
    threshold: float = typer.Option(
        0.8,
        help="Minimum threshold of confidence to retag a model. Default: 0.8"
    )
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
        num_proc=num_proc,
        batch_size=batch_size,
        tags_file_path=tags_file_path,
        threshold=threshold
    )
