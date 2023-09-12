import json
import logging
import random
import time

import typer
from loguru import logger

from datasets import Dataset, load_dataset, concatenate_datasets
from johnsnowlabs import nlp

import os

from sklearn.metrics import classification_report
import pyarrow.parquet as pq

from grants_tagger_light.utils.years_tags_parser import parse_years, parse_tags

import numpy as np

retag_app = typer.Typer()


def _load_data(dset: Dataset, tag, limit=100, split=0.8):
    """Load data from the IMDB dataset."""
    min_limit = min(len(dset), limit)
    dset = dset.select([x for x in range(limit)])
    # Not in parallel since the data is very small and it's worse to divide and conquer
    dset.map(
        lambda x: {'featured_tag': tag},
        desc=f"Adding featured tag ({tag})",
    )
    train_size = int(split * min_limit)
    train_dset = dset.select([x for x in range(train_size)])
    test_dset = dset.select([x for x in range(train_size, min_limit)])
    return train_dset, test_dset


def _create_pipelines(save_to_path, batch_size, train_df, test_df, tag, spark):
    """
        This method creates a Spark pipeline (to run on dataframes)
    Args:
        save_to_path: path where to save the final results.
        batch_size: max size of the batch to train. Since data is small for training, I limit it to 8.
        train_df: Spark Dataframe of the train data
        test_df: Spark Dataframe of the test data
        spark: the Spark Object

    Returns:
        a tuple of (pipeline, lightpipeline)
    """
    document_assembler = nlp.DocumentAssembler() \
        .setInputCol("abstractText") \
        .setOutputCol("document")

    # Biobert Sentence Embeddings (clinical)
    embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_biobert_clinical_base_cased", "en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentence_embeddings")

    retrain = True
    clf_dir = f"{save_to_path}.{tag.replace(' ', '')}_clf"
    if os.path.isdir(clf_dir):
        answer = input("Classifier already trained. Do you want to reuse it? [y|n]: ")
        while answer not in ['y', 'n']:
            answer = input("Classifier already trained. Do you want to reuse it? [y|n]: ")
        if answer == 'y':
            retrain = False

    if retrain:
        # I'm limiting the batch size to 8 since there are not many examples and big batch sizes will decrease accuracy
        classifierdl = nlp.ClassifierDLApproach() \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("label") \
            .setLabelColumn("featured_tag") \
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
        preds_df = preds.select('featured_tag', 'abstractText', 'label.result').toPandas()
        preds_df['result'] = preds_df['result'].apply(lambda x: x[0])
        logging.info(classification_report(preds_df['featured_tag'], preds_df['result']))

        logging.info("- Loading the model for prediction...")
        fit_clf_pipeline.stages[-1].write().overwrite().save(clf_dir)

    fit_clf_model = nlp.ClassifierDLModel.load(clf_dir)

    pred_pipeline = nlp.Pipeline(stages=[document_assembler,
                                         embeddings,
                                         fit_clf_model])
    pred_df = spark.createDataFrame([['']]).toDF("text")
    fit_pred_pipeline = pred_pipeline.fit(pred_df)

    return fit_pred_pipeline


def _annotate(save_to_path, dset, tag, limit, is_positive):
    human_supervision = {}
    curation_file = f"{save_to_path}.{tag.replace(' ', '')}.curation.json"
    if os.path.isfile(curation_file):
        with open(curation_file, 'r') as f:
            human_supervision = json.load(f)
        prompt = f"File `{curation_file}` found. Do you want to reuse previous work? [y|n]: "
        answer = input(prompt)
        while answer not in ['y', 'n']:
            answer = input(prompt)
        if answer == 'n':
            human_supervision[tag][is_positive] = []

    if tag not in human_supervision:
        human_supervision[tag] = {'positive': [], 'negative': []}

    field = 'positive' if is_positive else 'negative'
    count = len(human_supervision[tag][field])
    logging.info(f"[{tag}] Annotated: {count} Required: {limit} Available: {len(dset) - count}")
    finished = False
    while count <= limit:
        tries = 0
        random.seed(time.time())
        random_pos_row = random.randint(0, len(dset))
        id_ = dset[random_pos_row]['pmid']
        while id_ in [x['pmid'] for x in human_supervision[tag][field]]:
            random_pos_row = random.randint(0, len(dset))
            id_ = dset[random_pos_row]['pmid']
            tries += 1
            if tries >= 10:
                logger.error(f"Unable to find more examples for {field} {tag} which are not already tagged. "
                             f"Continuing with {count} examples...")
                finished = True
                break
        if finished:
            break
        print("="*50)
        print(dset[random_pos_row]['abstractText'])
        print("=" * 50)
        res = input(f'[{count}/{limit}]> Is this {"NOT " if not is_positive else ""} a `{tag}` text? '
                    f'[a to accept]: ')
        if res == 'a':
            human_supervision[tag][field].append(dset[random_pos_row])
            with open(curation_file, 'w') as f:
                json.dump(human_supervision, f)
        count = len(human_supervision[tag][field])


def _curate(save_to_path, pos_dset, neg_dset, tag, limit):
    logging.info("- Curating positive examples")
    _annotate(save_to_path, pos_dset, tag, limit, is_positive=True)

    logging.info("- Curating negative examples")
    _annotate(save_to_path, neg_dset, tag, limit, is_positive=False)


def retag(
    data_path: str,
    save_to_path: str,
    spark_memory: int = 27,
    num_proc: int = os.cpu_count(),
    batch_size: int = 64,
    tags: list = None,
    tags_file_path: str = None,
    threshold: float = 0.8,
    train_examples: int = 100,
    supervised: bool = True,
    years: list = None,
):

    spark = nlp.start(spark_conf={
        'spark.driver.memory': f'{spark_memory}g',
        'spark.executor.memory': f'{spark_memory}g',
    })

    # We only have 1 file, so no sharding is available https://huggingface.co/docs/datasets/loading#multiprocessing
    logging.info("Loading the MeSH jsonl...")
    dset = load_dataset("json", data_files=data_path, num_proc=1)
    if "train" in dset:
        dset = dset["train"]

    if years is not None:
        logger.info(f"Removing all years which are not in {years}")
        dset = dset.filter(
            lambda x: any(np.isin(years, [str(x["year"])])), num_proc=num_proc
        )

    if tags_file_path is not None and os.path.isfile(tags_file_path):
        with open(tags_file_path, 'r') as f:
            tags = [x.strip() for x in f.readlines()]

    logging.info(f"Total tags detected: {tags}")

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

        logging.info(f"- Obtaining negative examples ('other') for {tag}...")
        negative_dset = dset.filter(
            lambda x: tag not in x["meshMajor"], num_proc=num_proc
        )

        if supervised:
            logging.info(f"- Curating data...")
            _curate(save_to_path, positive_dset, negative_dset, tag, train_examples)

            curation_file = f"{save_to_path}.{tag.replace(' ', '')}.curation.json"
            if os.path.isfile(curation_file):
                with open(curation_file, "r") as fr:
                    # I load the curated data file
                    human_supervision = json.load(fr)
                    positive_dset = Dataset.from_list(human_supervision[tag]['positive'])
                    negative_dset = Dataset.from_list(human_supervision[tag]['negative'])

        pos_x_train, pos_x_test = _load_data(positive_dset, tag, limit=train_examples, split=0.8)
        neg_x_train, neg_x_test = _load_data(negative_dset, "other", limit=train_examples, split=0.8)

        pos_x_train = pos_x_train.add_column("featured_tag", [tag] * len(pos_x_train))
        pos_x_test = pos_x_test.add_column("featured_tag", [tag] * len(pos_x_test))
        neg_x_train = neg_x_train.add_column("featured_tag", ["other"] * len(neg_x_train))
        neg_x_test = neg_x_test.add_column("featured_tag", ["other"] * len(neg_x_test))

        logging.info(f"- Creating train/test sets...")
        train = concatenate_datasets([pos_x_train, neg_x_train])
        train_df = spark.createDataFrame(train)
        test = concatenate_datasets([pos_x_test, neg_x_test])
        test_df = spark.createDataFrame(test)

        logging.info(f"- Train dataset size: {train_df.count()}")
        logging.info(f"- Test dataset size: {test_df.count()}")

        logging.info(f"- Creating `sparknlp` pipelines...")
        pipeline = _create_pipelines(save_to_path, batch_size, train_df, test_df, tag, spark)

        logging.info(f"- Optimizing dataframe...")
        data_in_parquet = f"{save_to_path}.data.parquet"
        optimize=True
        if os.path.isfile(data_in_parquet):
            answer = input("Optimized dataframe found. Do you want to use it? [y|n]: ")
            while answer not in ['y', 'n']:
                answer = input("Optimized dataframe found. Do you want to use it? [y|n]: ")
            if answer == 'y':
                optimize = False

        if optimize:
            dset = dset.remove_columns(["title", "journal", "year"])

            pq.write_table(dset.data.table, data_in_parquet)
        del dset, train, train_df, test, test_df, pos_x_train, pos_x_test, neg_x_train, neg_x_test, positive_dset,\
            negative_dset
        sdf = spark.read.load(data_in_parquet)

        logging.info(f"- Repartitioning...")
        sdf = sdf.repartition(num_proc)

        logging.info(f"- Retagging {tag}...")
        pipeline.transform(sdf).write.mode('overwrite').save(f"{save_to_path}.{tag.replace(' ', '')}.prediction")

        # 1) We load
        # 2) We filter to get those results where the predicted tag was not initially in meshMajor
        # 3) We filter by confidence > threshold
        # predictions = spark.read.load(f"{save_to_path}.{tag}.prediction").\
        #   filter(~array_contains(col('meshMajor'), tag)).\


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
    tags: str = typer.Option(
        None,
        help="Comma separated list of tags to retag"
    ),
    tags_file_path: str = typer.Option(
        None,
        help="Text file containing one line per tag to be considered. "
        "The rest will be discarded.",
    ),
    threshold: float = typer.Option(
        0.9,
        help="Minimum threshold of confidence to retag a model. Default: 0.9"
    ),
    train_examples: int = typer.Option(
        100,
        help="Number of examples to use for training the retaggers"
    ),
    supervised: bool = typer.Option(
        True,
        help="Use human curation, showing a `limit` amount of positive and negative examples to curate data"
             " for training the retaggers. The user will be required to accept or reject. When the limit is reached,"
             " the model will be train. All intermediary steps will be saved."
    ),
    spark_memory: int = typer.Option(
        20,
        help="Gigabytes of memory to be used. Recommended at least 20 to run on MeSH."
    ),
    years: str = typer.Option(
        None, help="Comma-separated years you want to include in the retagging process"
    ),
):
    if not data_path.endswith("jsonl"):
        logger.error(
            "It seems your input MeSH data is not in `jsonl` format. "
            "Please, run first `scripts/mesh_json_to_jsonl.py.`"
        )
        exit(-1)

    if tags_file_path is None and tags is None:
        logger.error(
            "To understand which tags need to be augmented, use --tags [tags separated by comma] or create a file with"
            "a newline per tag and set the path in --tags-file-path"
        )
        exit(-1)

    if tags_file_path is not None and not os.path.isfile(tags_file_path):
        logger.error(
            f"{tags_file_path} not found"
        )
        exit(-1)

    retag(
        data_path,
        save_to_path,
        spark_memory=spark_memory,
        num_proc=num_proc,
        batch_size=batch_size,
        tags=parse_tags(tags),
        tags_file_path=tags_file_path,
        threshold=threshold,
        train_examples=train_examples,
        supervised=supervised,
        years=parse_years(years),
    )
