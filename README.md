# Grants Tagger Light 🔖
Light weight repository for grant tagger model deployment and inference.
Adapted from [the original repository](https://github.com/wellcometrust/grants_tagger)

Grants tagger is a machine learning powered tool that
assigns biomedically related tags to grants proposals.
Those tags can be custom to the organisation
or based upon a preexisting ontology like MeSH.

The tool is current being developed internally at the
Wellcome Trust for internal use but both the models and the
code will be made available in a reusable manner.

This work started as a means to automate the tags of one
funding division within Wellcome but currently it has expanded
into the development and automation of a complete set of tags
that can cover past and future directions for the organisation.

Science tags refer to the custom tags for the Science funding
division. These tags are higly specific to the research Wellcome
funds so it is not advisable to use them.

MeSH tags are subset of tags from the MeSH ontology that aim to
tags grants according to:
- diseases
- themes of research
Those tags are generic enough to be used by other biomedical funders
but note that the selection of tags are highly specific to Wellcome
at the moment.

# 💻 Installation

## 0. Install poetry
`curl -sSL https://install.python-poetry.org | python3 -`

## 1. Install dependencies
For CPU-support:
`poetry install`

For GPU-support:
`poetry install --with gpu`

For training the model, we recommend installing the version of this package with GPU support.
For infenrece, CPU-support should suffice.

## 2. Activate the environment
`poetry shell`

You now have access to the `grants-tagger` command line interface!


# ⌨️  Commands

| Commands        |                                                              | needs dev |
| --------------- | ------------------------------------------------------------ | --------- |
| ⚙️  preprocess   | preprocess data to use for training                          | False |
| 🔥 train        | trains a new model                                           | True |
| 📈 evaluate     | evaluate performance of pretrained model                     | True |
| 🔖 predict      | predict tags given a grant abstract using a pretrained model | False |
| 🎛 tune         | tune params and threshold                                    | True |
| ⬇️  download    | download data from EPMC                                      | False |

in square brackets the commands that are not implemented yet

## ⚙️  Preprocess

Preprocess creates a JSONL datafile with `text`, `tags` and `meta` as keys.
Text and tags are used for training whereas meta can be useful during annotation
or to analyse predictions and performance. Each dataset needs its own
preprocessing so the current preprocess works with the bioasq-mesh one.
If you want to use a different dataset see section on bringing
your own data under development.


#### bioasq-mesh
```

 Usage: grants-tagger preprocess bioasq-mesh [OPTIONS] [INPUT_PATH]
                                             [TRAIN_OUTPUT_PATH]
                                             [LABEL_BINARIZER_PATH]

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   input_path                [INPUT_PATH]            path to BioASQ JSON data [default: None]                                                                           │
│   train_output_path         [TRAIN_OUTPUT_PATH]     path to JSONL output file that will be generated for the train set [default: None]                                 │
│   label_binarizer_path      [LABEL_BINARIZER_PATH]  path to pickle file that will contain the label binarizer [default: None]                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --test-output-path        TEXT     path to JSONL output file that will be generated for the test set [default: None]                                                   │
│ --mesh-tags-path          TEXT     path to mesh tags to filter [default: None]                                                                                         │
│ --test-split              FLOAT    split percentage for test data. if None no split. [default: 0.01]                                                                   │
│ --filter-years            TEXT     years to keep in form min_year,max_year with both inclusive [default: None]                                                         │
│ --config                  PATH     path to config files that defines arguments [default: None]                                                                         │
│ --n-max                   INTEGER  Maximum limit on the number of datapoints in the set (including training and test) [default: None]                                  │
│ --help                             Show this message and exit.                                                                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 🔥 Train

Train acts as the entry point command for training all models. Currently we only support
the BertMesh model. The command will train a model and save it to the specified path.

### bertmesh
```

 Usage: grants-tagger train bertmesh [OPTIONS] MODEL_KEY DATA_PATH
                                     MODEL_SAVE_PATH

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_key            TEXT  Pretrained model key. Local path or HF location [default: None] [required]                                                                          │
│ *    data_path            TEXT  Path to data in jsonl format. Must contain text and tags field [default: None] [required]                                                           │
│ *    model_save_path      TEXT  Path to save model to [default: None] [required]                                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                         │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```


## 📈 Evaluate

Evaluate enables evaluation of the performance of various approaches including
human performance and other systems like MTI, SciSpacy and soon Dimensions. As
such evaluate has the followin subcommands

### model

Model is the generic entrypoint for model evaluation. Similar to train approach
controls which model will be evaluated. Approach which is a positional argument
in this command controls which model will be evaluated. Since the data in train
are sometimes split inside train, the same splitting is performed in evaluate.
Evaluate only supports some models, in particular those that have made it to
production. These are: `tfidf-svm`, `scibert`, `science-ensemble`, `mesh-tfidf-svm`
and `mesh-cnn`. Note that train also outputs evaluation scores so for models
not made into production this is the way to evaluate. The plan is to extend
evaluate to all models when train starts training explicit model approaches.

```

 Usage: grants-tagger evaluate model [OPTIONS] MODEL_PATH DATA_PATH

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_path      TEXT  comma separated paths to pretrained models [default: None] [required]                                                                                    │
│ *    data_path       PATH  path to data that was used for training [default: None] [required]                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --threshold                              TEXT  threshold or comma separated thresholds used to assign tags [default: 0.5]                                                           │
│ --results-path                           TEXT  path to save results [default: None]                                                                                                 │
│ --full-report-path                       TEXT  Path to save full report, i.e. more comprehensive results than the ones saved in results_path [default: None]                        │
│ --split-data          --no-split-data          flag on whether to split data in same way as was done in train [default: split-data]                                                 │
│ --config                                 PATH  path to config file that defines arguments [default: None]                                                                           │
│ --help                                         Show this message and exit.                                                                                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### grants
Evaluate an xlinear model on grants data.
```

 Usage: grants-tagger evaluate grants [OPTIONS] MODEL_PATH DATA_PATH
                                      LABEL_BINARIZER_PATH

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_path                TEXT  comma separated paths to pretrained models [default: None] [required]                                                                 │
│ *    data_path                 PATH  path to data that was used for training [default: None] [required]                                                                    │
│ *    label_binarizer_path      PATH  path to label binarize [default: None] [required]                                                                                     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --threshold                            TEXT  threshold or comma separated thresholds used to assign tags [default: 0.5]                                                    │
│ --results-path                         TEXT  path to save results [default: None]                                                                                          │
│ --mesh-tags-path                       TEXT  path to mesh subset to evaluate [default: None]                                                                               │
│ --parameters        --no-parameters          stringified parameters for model evaluation, if any [default: no-parameters]                                                  │
│ --config                               PATH  path to config file that defines arguments [default: None]                                                                    │
│ --help                                       Show this message and exit.                                                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 🔖 Predict

Predict assigns tags on a given abstract text that you can pass as argument.


```

 Usage: grants-tagger predict [OPTIONS] TEXT MODEL_PATH

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    text            TEXT  [default: None] [required]                                                                                                                      │
│ *    model_path      PATH  [default: None] [required]                                                                                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --batch-size                             INTEGER  [default: 1]                                                                                                             │
│ --probabilities    --no-probabilities             [default: no-probabilities]                                                                                              │
│ --threshold                              FLOAT    [default: 0.5]                                                                                                           │
│ --help                                            Show this message and exit.                                                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## 🎛 Tune
Optimise the threshold used for tag decisions.

### threshold
```

 Usage: grants-tagger tune threshold [OPTIONS] DATA_PATH MODEL_PATH
                                     LABEL_BINARIZER_PATH THRESHOLDS_PATH

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path                 PATH  path to data in jsonl to train and test model [default: None] [required]                                                              │
│ *    model_path                PATH  path to data in jsonl to train and test model [default: None] [required]                                                              │
│ *    label_binarizer_path      PATH  path to label binarizer [default: None] [required]                                                                                    │
│ *    thresholds_path           PATH  path to save threshold values [default: None] [required]                                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --val-size                             FLOAT    validation size of text data to use for tuning [default: 0.8]                                                              │
│ --nb-thresholds                        INTEGER  number of thresholds to be tried divided evenly between 0 and 1 [default: None]                                            │
│ --init-threshold                       FLOAT    initial threshold value to compare against [default: 0.2]                                                                  │
│ --split-data        --no-split-data             flag on whether to split data as was done for train [default: no-split-data]                                               │
│ --help                                          Show this message and exit.                                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## ⬇️  Download

This commands enables you to download mesh data from EPMC

### epmc-mesh

```

 Usage: grants-tagger download epmc-mesh [OPTIONS] DOWNLOAD_PATH

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    download_path      TEXT  path to directory where to download EPMC data [default: None] [required]                                                                     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --year        INTEGER  year to download epmc publications [default: 2020]                                                                                                  │
│ --help                 Show this message and exit.                                                                                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

# 🧑🏻‍💻  Develop

Install development dependencies via:
`poetry install --with dev`

## 📋 Env variables

If you want to participate to BIOASQ competition you need to set some variables.

Variable              | Required for       | Description
--------------------- | ------------------ | ----------
BIOASQ_USERNAME       | bioasq             | username with which registered in BioASQ
BIOASQ_PASSWORD       | bioasq             | password            --//--

If you use [direnv](https://direnv.net) then you can use it to populate
your `.envrc` which will export the variables automatically, otherwise
ensure you export every time or include in your bash profile.


## ✔️  Reproduce

To reproduce production models we use DVC. DVC defines a directed
acyclic graph (DAG) of steps that need to run to reproduce a model
or result. You can see all steps with `dvc dag`. You can reproduce
all steps with `dvc repro`. You can reproduce any step of the DAG
with `dvc repro STEP_NAME` for example `dvc repro train_tfidf_svm`.
Note that mesh models require a GPU to train and depending on the
parameters it might take from 1 to several days.

You can reproduce individual experiments using one of the configs in
the dedicated `/configs` folder. You can run all steps of the pipeline
using `./scripts/run_DATASET_config.sh path_to_config` where DATASET
can be one of science or mesh. You can also run individual steps
with the CLI commands e.g. `grants_tagger preprocess bioasq-mesh --config path_to_config`
and `grants_tagger train --config path_to_config`.

## 💾 Bring your own data

To use grants_tagger with your own data the main thing you need to
implement is a new preprocess function that creates a JSONL with the
fields `text`, `tags` and `meta`. Meta can be even left empty if you
do not plan to use it. You can easily plug the new preprocess into the
cli by importing your function to `grants_tagger/cli.py` and
define the subcommand name for your preprocess. For example if the
function was preprocessing EPMC data for MESH it could be
```
@preprocess_app.command()
def epmc_mesh(...)
```
and you would be able to run `grants_tagger preprocess epmc_mesh ...`

## 🚦 Test

Run tests with `pytest`. If you want to write some additional tests,
they should go in the subfolde `tests/`


## ✍️ Scripts

Additional scripts, mostly related to Wellcome Trust-specific code can be
found in `/scripts`. Please refer to the [readme](scripts/README.md) therein for more info
on how to run those.

To install dependencies for the scripts, simply run:
`poetry install --with scripts`
