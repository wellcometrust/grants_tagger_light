# Grants Tagger Light 🔖
Lightweight repository for grant tagger model deployment and inference.
Adapted from [the original repository](https://github.com/wellcometrust/grants_tagger)

Grants tagger is a machine learning powered tool that
assigns biomedical-related tags to grant proposals.
Those tags can be custom to the organisation
or based upon a preexisting ontology like MeSH.

The tool is current being developed internally at the
Wellcome Trust for internal use but both the models and the
code will be made available in a reusable manner.

This work started as a means to automate the tags of one
funding division within Wellcome, but currently it has expanded
into the development and automation of a complete set of tags
that can cover past and future directions for the organisation.

Science tags refer to the custom tags for the Science funding
division. These tags are highly specific to the research Wellcome
funds, so it is not advisable to use them.

MeSH tags are subset of tags from the MeSH ontology that aim to
tag grants according to:
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
For inference, CPU-support should suffice.

## 2. Activate the environment
`poetry shell`

You now have access to the `grants-tagger` command line interface!

## 3. For `retagging` you will need to make sure you have `openjdk 8 (or 11)` installed to run Spark
First, make sure you don't have java installed or you have another version that it's not java 8 or 11.
```shell
java -version
```

If you don't or you have another version, install it (example for java 8):
```shell
sudo apt update
sudo apt install openjdk-8-jdk
```

Make sure you set by default the one we have just installed. Copy the path to the java folder from:
```shell
sudo update-alternatives --config java
```

And now, set your JAVA_HOME env var:
```shell
sudo vim /etc/environment
JAVA_HOME="[PATH_TO_THE_JAVA_FOLDER]
```

Restar the shell or do `source /etc/environment`


## OPTIONAL: 4. Install MantisNLP `remote` to connect to a remote AWS instances
`pip install git+https://github.com/ivyleavedtoadflax/remote.py.git`
Then add your instance
`remote config add [instance_name]`
And then connect and attach to your machine with a tunnel
`remote connect -p 1234:localhost:1234 -v`

# ⌨️  Commands

| Commands     | Description                                                  | Needs dev |
|--------------|--------------------------------------------------------------|-----------|
| ⚙ preprocess | preprocess and save the data outside training                | False     |
| 🔥 train     | preprocesses the data and trains a new model                 | True      |
| 📚 augment   | augments data using an LLM (gpt)                             | False     |
| ✏ retag      | retags data using XLinear to correct errors                  | False     |
| 📈 evaluate  | evaluate performance of pretrained model                     | True      |
| 🔖 predict   | predict tags given a grant abstract using a pretrained model | False     |
| 🎛 tune      | tune params and threshold                                    | True      |
| ⬇ download   | download data from EPMC                                      | False     |


in square brackets the commands that are not implemented yet

## ⚙️Preprocess

This process is optional to run, since it can be directly managed by the `Train` process.
- If you run it manually, it will store the data in local first, which can help if you need finetune in the future, 
rerun, etc.
- If not run it, the `train` step will preprocess and then run, without any extra I/O operations on disk, 
which may add latency depending on the infrastructure.

It requires data in `jsonl` format for parallelization purposes. In `data/raw` you can find `allMesH_2021.jsonl` 
already prepared for the preprocessing step.

If your data is in `json` format, trasnform it to `jsonl` with tools as `jq` or using Python.
You can use an example of `allMeSH_2021.json` conversion to `jsonl` in `scripts/mesh_json_to_jsonl.py`:

```bash
python scripts/mesh_json_to_jsonl.py --input_path data/raw/allMeSH_2021.json --output_path data/raw/test.jsonl --filter_years 2020,2021
```

Each dataset needs its own preprocessing so the current preprocess works with the `allMeSH_2021.jsonl` one.

If you want to use a different dataset see section on bringing
your own data under development.


### Preprocessing bertmesh

```
 Usage: grants-tagger preprocess mesh [OPTIONS] DATA_PATH SAVE_TO_PATH                                                                                                                                             
                                      MODEL_KEY                                                                                                                                                                    
                                                                                                                                                                                                                   
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path         TEXT  Path to mesh.jsonl [default: None] [required]                                                                                                                                      │
│ *    save_to_path      TEXT  Path to save the serialized PyArrow dataset after preprocessing [default: None] [required]                                                                                         │
│ *    model_key         TEXT  Key to use when loading tokenizer and label2id. Leave blank if training from scratch [default: None] [required]                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --test-size          FLOAT    Fraction of data to use for testing in (0,1] or number of rows [default: None]                                                                                                    │
│ --num-proc           INTEGER  Number of processes to use for preprocessing [default: 8]                                                                                                                         │
│ --max-samples        INTEGER  Maximum number of samples to use for preprocessing [default: -1]                                                                                                                  │
│ --batch-size         INTEGER  Size of the preprocessing batch [default: 256]                                                                                                                                    │
│ --tags               TEXT     Comma-separated tags you want to include in the dataset (the rest will be discarded) [default: None]                                                                              │
│ --train-years        TEXT     Comma-separated years you want to include in the training dataset [default: None]                                                                                                 │
│ --test-years         TEXT     Comma-separated years you want to include in the test dataset [default: None]                                                                                                     │
│ --help                        Show this message and exit.                                                                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 🔥 Train

The command will train a model and save it to the specified path. Currently we support on BertMesh.

### Training bertmesh
```
 Usage: grants-tagger train bertmesh [OPTIONS] MODEL_KEY DATA_PATH                                                                                                                                                 

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_key      TEXT  Pretrained model key. Local path or HF location [default: None] [required]                                                                                                            │
│ *    data_path      TEXT  Path to allMeSH_2021.jsonl (or similar) or to a folder after preprocessing and saving to disk [default: None] [required]                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --test-size              FLOAT    Fraction of data to use for testing (0,1] or number of rows [default: None]                                                                                                   │
│ --num-proc               INTEGER  Number of processes to use for preprocessing [default: 8]                                                                                                                     │
│ --max-samples            INTEGER  Maximum number of samples to use from the json [default: -1]                                                                                                                  │
│ --shards                 INTEGER  Number os shards to divide training IterativeDataset to (improves performance) [default: 8]                                                                                   │
│ --from-checkpoint        TEXT     Name of the checkpoint to resume training [default: None]                                                                                                                     │
│ --tags                   TEXT     Comma-separated tags you want to include in the dataset (the rest will be discarded) [default: None]                                                                          │
│ --train-years            TEXT     Comma-separated years you want to include in the training dataset [default: None]                                                                                             │
│ --test-years             TEXT     Comma-separated years you want to include in the test dataset [default: None]                                                                                                 │
│ --help                            Show this message and exit.                                                                                                                                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### About `model_key`
`model_key` possible values are:
- A HF location for a pretrained / finetuned model 
- "" to load a model by default and train from scratch (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)

#### About `sharding`
`sharding` was proposed by [Hugging Face](https://github.com/huggingface/datasets/issues/2252#issuecomment-825596467)
to improve performance on big datasets. To enable it:
- set shards to something bigger than 1 (Recommended: same number as cpu cores)

#### Other arguments
Besides those arguments, feel free to add any other TrainingArgument from Hugging Face or Wand DB. 
This is the example used to train reaching a ~0.6 F1, also available at `examples/train_by_epochs.sh`
```shell
grants-tagger train bertmesh \
    "" \
    [YOUR_PREPROCESSED_FOLDER] \
    --output_dir [YOUR_OUTPUT_FOLDER] \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --multilabel_attention True \
    --freeze_backbone unfreeze \
    --num_train_epochs 7 \
    --learning_rate 5e-5 \
    --dropout 0.1 \
    --hidden_size 1024 \
    --warmup_steps 5000 \
    --max_grad_norm 2.0 \
    --scheduler_type cosine_hard_restart \
    --weight_decay 0.2 \
    --correct_bias True \
    --threshold 0.25 \
    --prune_labels_in_evaluation True \
    --hidden_dropout_prob 0.2 \
    --attention_probs_dropout_prob 0.2 \
    --fp16 \
    --torch_compile \
    --evaluation_strategy epoch \
    --eval_accumulation_steps 20 \
    --save_strategy epoch \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY}
```

## 📚 Augment
Data augmentation can be useful for low represented classes. LLMs as `openai GPT-3.5` can be used to that purpose.

### Augmenting bertmesh
For bertmesh, we will augment the `allMeSH_2021.jsonl` file. We just need to select the path to that file (usually in `data/raw/allMeSH_2021.jsonl`)
and where to save the generated data (in jsonl).

```shell
grants-tagger augment mesh [FOLDER_AFTER_PREPROCESSING] [SET_YOUR_OUTPUT_FOLDER_HERE] \
```

### concurrent-calls param
By setting `concurrent-calls [number_of_calls]` you will use the multiclient openai library which will create 
async calls to openai and work in parallel, improving the processing times.

If `1`, vanilla `openai` library in sync mode will be used.

### What tags do we augment? By minimum examples 
There are two ways to do it. First, `all tags` with less than `min-examples` examples.
In this case, There are two parameters which are important to know:
* `min-examples`: Example: 25. Is the min. number of examples you require from a tag. If less is found, the data augmentation will be triggered.
* `examples`: Example: 25. In case there are less than `min-examples`, how many examples we generate for that tag.

```shell
grants-tagger augment mesh [FOLDER_AFTER_PREPROCESSING] [SET_YOUR_OUTPUT_FOLDER_HERE] \
  --min-examples 25 \
  --concurrent-calls 25
```

### What tags do we augment? By tags file
Second way is to use a file with 1 line per tag. To do this, instead of `min-examples` use `tags-file-path` param.
```shell
grants-tagger augment mesh [FOLDER_AFTER_PREPROCESSING] [SET_YOUR_OUTPUT_FOLDER_HERE] \
  --tags-file-path tags_to_augment.txt \
  --examples 25 \
  --concurrent-calls 25
```

### Other params
```                                                                                                                                                                                                                   
 Usage: grants-tagger augment mesh [OPTIONS] DATA_PATH SAVE_TO_PATH                                                                                                                                                

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path         TEXT  Path to mesh.jsonl [default: None] [required]                                                                                                                                      │
│ *    save_to_path      TEXT  Path to save the new generated data in jsonl format
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --model-key               TEXT                   LLM to use data augmentation. By now, only `openai` is supported [default: gpt-3.5-turbo]                                                                      │
│ --num-proc                INTEGER                Number of processes to use for data augmentation [default: 8]                                                                                                  │
│ --batch-size              INTEGER                Preprocessing batch size (for dataset, filter, map, ...) [default: 64]                                                                                         │
│ --min-examples            INTEGER                Minimum number of examples to require. Less than that will trigger data augmentation. [default: None]                                                          │
│ --examples                INTEGER                Examples to generate per each tag. [default: 25]                                                                                                               │
│ --prompt-template         TEXT                   File to use as a prompt. Make sure to ask the LLM to return a dict with two fields: `abstract` and `tags`                                                      │
│                                                  [default: grants_tagger_light/augmentation/prompt.template]                                                                                                    │
│ --concurrent-calls        INTEGER RANGE [x>=1]   Concurrent calls with 1 tag each to the different model [default: 16]                                                                                          │
│ --temperature             FLOAT RANGE [0<=x<=2]  A value between 0 and 2. The bigger - the more creative. [default: 1.5]                                                                                        │
│ --tags-file-path          TEXT                   Text file containing one line per tag to be considered. The rest will be discarded. [default: None]                                                            │
│ --help                                           Show this message and exit.                                                                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## ✏ Retagging
Retagging is the process of correcting inconsistent tags in the data.

## Retagging bertmesh
The data in `allMeSH_2021.jsonl` (`PubMed` labelled with `MeSH` tags) is highly inconsistent for many rows, leading to 
bad performance of some ambiguous labels. 

Example: this is a row not being tagged as `Artificial Intelligence`, but talking about `Neural Networks`.

```python
{"journal": "Nature communications", "meshMajor": ["Cell Cycle", "Image Processing, Computer-Assisted", "Microscopy", "Neural Networks, Computer", "Saccharomyces cerevisiae", "Software"], "year": "2020", "abstractText": "The identification of cell borders ('segmentation') in microscopy images constitutes a bottleneck for large-scale experiments. For the model organism Saccharomyces cerevisiae, current segmentation methods face challenges when cells bud, crowd, or exhibit irregular features. We present a convolutional neural network (CNN) named YeaZ, the underlying training set of high-quality segmented yeast images (>10 000 cells) including mutants, stressed cells, and time courses, as well as a graphical user interface and a web application ( www.quantsysbio.com/data-and-software ) to efficiently employ, test, and expand the system. A key feature is a cell-cell boundary test which avoids the need for fluorescent markers. Our CNN is highly accurate, including for buds, and outperforms existing methods on benchmark images, indicating it transfers well to other conditions. To demonstrate how efficient large-scale image processing uncovers new biology, we analyze the geometries of ?2200 wild-type and cyclin mutant cells and find that morphogenesis control occurs unexpectedly early and gradually.", "pmid": "33184262", "title": "A convolutional neural network segments yeast microscopy images with high accuracy."}]
```

And this is another example. Same topic, but now it was tagged as `Artificial Intelligence`.
```
{"journal": "Nature communications", "meshMajor": ["Databases, Factual", "Deep Learning", "Diagnosis, Computer-Assisted", "False Positive Reactions", "Humans", "Image Processing, Computer-Assisted", "Neural Networks, Computer", "Stomach Neoplasms"], "year": "2020", "abstractText": "The early detection and accurate histopathological diagnosis of gastric cancer increase the chances of successful treatment. The worldwide shortage of pathologists offers a unique opportunity for the use of artificial intelligence assistance systems to alleviate the workload and increase diagnostic accuracy. Here, we report a clinically applicable system developed at the Chinese PLA General Hospital, China, using a deep convolutional neural network trained with 2,123 pixel-level annotated H&E-stained whole slide images. The model achieves a sensitivity near 100% and an average specificity of 80.6% on a real-world test dataset with 3,212 whole slide images digitalized by three scanners. We show that the system could aid pathologists in improving diagnostic accuracy and preventing misdiagnoses. Moreover, we demonstrate that our system performs robustly with 1,582 whole slide images from two other medical centres. Our study suggests the feasibility and benefits of using histopathological artificial intelligence assistance systems in routine practice scenarios.", "pmid": "32855423", "title": "Clinically applicable histopathological diagnosis system for gastric cancer detection using deep learning."}
```

For tags as `Data Science`, `Artificial Intelligence`, `Data Collection`, `Deep Learning`, `Neural Networks, Computer`, `Machine Learning`, the situation is really dramatic.

`Artificial Intelligence` with several thousand rows shows a performance of 0.1 F1, showing a lot of confusion with the other tags described above.

We propose a solution: retagging the original data with a small curated dataset of examples and a quick Machine Learning light classifier: XLinear.

```
grants-tagger retag mesh data/raw/allMeSH_2021.jsonl [SET_YOUR_OUTPUT_FILE_HERE] \
  --tags "Artificial Intelligence,HIV" \
  --years 2016,2017,2018,2019,2020,2021 \
  --train-examples 100 \
  --batch-size 10000 \
  --supervised 
```
Let's take a look at some of the params:
- *tags*: A comma-separated (and quoted) list of tags you want to retag.
- *years*: A comma-separated list of years you want to include
- *train-examples*: The number of examples to include for training the classifier. Default: 100
- *batch-size*: The size of the processing batch. Keep it high as the memory consumption is really small. Default: 10000

### Getting the curation data: Supervised or Unsupervised?
For using the retagger, you need a small 
- *supervised*: If you want to be asked for *train-examples* examples to curate a dataset for training the classifier. Recommended.

```
==================================================
The SD BIOLINE HIV/Syphilis Duo assay is the first World Health Organization prequalified dual rapid diagnostic test for
 simultaneous detection of HIV and Treponema pallidum antibodies in human blood. Prior to introducing the test into 
 antenatal clinics across South Sudan, a field evaluation of its clinical performance in diagnosing both HIV and 
 syphilis in pregnant women was conducted. SD Bioline test performance on venous blood samples was compared with (i) 
 Vironostika HIV1/2 Uniform II Ag/Ab reference standard and Alere Determine HIV 1/2 non-reference standard for HIV 
 diagnosis, and (ii) Treponema pallidum hemagglutination reference standard and Rapid plasma reagin non-reference 
 standard for syphilis. Sensitivity, specificity, positive predictive value (PPN), negative predictive value (NPV) 
 and kappa (ê) value were calculated for each component against the reference standards within 95% confidence 
 intervals (CIs); agreements between Determine HIV 1/2 and SD Bioline HIV tests were also calculated. Of 442 pregnant 
 women recruited, eight (1.8%) were HIV positive, 22 (5.0%) had evidence of syphilis exposure; 14 (3.2%) had active 
 infection. For HIV diagnosis, the sensitivity, specificity, PPV and NPV were 100% (95% CI: 63.1-100), 100% 
 (95% CI: 99.2-100), 100% (95% CI: 63.1-100) and 100% (95% CI: 99.2-100) respectively with ê value of 1 
 (95% CI: 0.992-1.000). Overall agreement of the Duo HIV component and Determine test was 99.1% (95% CI: 0.977-0.998) 
 with 66.7% (95% CI: 34.9-90.1) positive and 100% (95% CI: 0.992-1.000) negative percent agreements. For syphilis, 
 the Duo assay sensitivity was 86.4% (95% CI: 65.1-97.1) and specificity 100% (95% CI: 99.1-100) with PPV 100% 
 (95% CI: 82.4-100), NPV 99.2% (95% CI: 97.9-99.9) and ê value 0.92 (95% CI: 0.980-0.999). Our findings suggest the SD Bioline HIV/Syphilis Duo Assay could be suitable for HIV and syphilis testing in women attending antenatal services across South Sudan. Women with positive syphilis results should receive treatment immediately, whereas HIV positive women should undergo confirmatory testing following national HIV testing guidelines.
==================================================
[2/100]> Is this  a `HIV` text? [a to accept]:

```

If not set, the model will randomly get *train-examples* and train the classifier without your supervision, which will reduce the performance of the classifiers.

### Artifacts created
As a result of the proces, you will find a folder at *save_to_path*. Inside,  you will find:
- One folder per tag, including:
  - `clf` (a classifier),
  - `curation` (a dataset of positive and negative examples for the tag)
  - `labelbinarizer` (a label binarizer to encode the labels)  
- a `corrections` file, the new allMeSH_2021.jsonl with your tags corrected.

### Other params
```
 Usage: grants-tagger retag mesh [OPTIONS] DATA_PATH SAVE_TO_PATH                                                                                                                                                  

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path         TEXT  Path to mesh.jsonl [default: None] [required]                                                                                                                                      │
│ *    save_to_path      TEXT  Path where to save the retagged data [default: None] [required]                                                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --num-proc                             INTEGER  Number of processes to use for data augmentation [default: 8]                                                                                                   │
│ --batch-size                           INTEGER  Preprocessing batch size (for dataset, filter, map, ...) [default: 64]                                                                                          │
│ --tags-file-path                       TEXT     Text file containing one line per tag to be considered. The rest will be discarded. [default: None]                                                             │
│ --threshold                            FLOAT    Minimum threshold of confidence to retag a model. Default: 0.9 [default: 0.9]                                                                                   │
│ --train-examples                       INTEGER  Number of examples to use for training the retaggers [default: 100]                                                                                             │
│ --supervised        --no-supervised             Use human curation, showing a `limit` amount of positive and negative examples to curate data for training the retaggers. The user will be required to accept   │
│                                                 or reject. When the limit is reached, the model will be train. All intermediary steps will be saved.                                                            │
│                                                 [default: supervised]                                                                                                                                           │
│ --help                                          Show this message and exit.                                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
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

The project has references to `dvc` big files. You can just do `dvc pull` and retrieve those,
including `allMeSH_2021.json`  and `allMeSH_2021.jsonl` to train `bertmesh`.

Also, this commands enables you to download mesh data from EPMC

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

Variable              | Required for | Description
--------------------- |--------------| ----------
WANDB_API_KEY         | train        | key to dump the results to Weights&Biases
AWS_ACCESS_KEY_ID     | train        | access key to pull data from dvc on S3
AWS_SECRET_ACCESS_KEY | train        | secret key to pull data from dvc on S3

If you want to participate to BIOASQ competition you need to set some variables.

Variable              | Required for       | Description
--------------------- | ------------------ | ----------
BIOASQ_USERNAME       | bioasq             | username with which registered in BioASQ
BIOASQ_PASSWORD       | bioasq             | password

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

To run the test you need to have installed the `dev` dependencies first. 
This is done by running `poetry install --with dev` after you are in the sell (`poetry shell`)

Run tests with `pytest`. If you want to write some additional tests,
they should go in the subfolde `tests/`


## ✍️ Scripts

Additional scripts, mostly related to Wellcome Trust-specific code can be
found in `/scripts`. Please refer to the [readme](scripts/README.md) therein for more info
on how to run those.

To install dependencies for the scripts, simply run:
`poetry install --with scripts`
