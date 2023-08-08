# Run on g5.12xlargeinstance

# Without preprocessing (on-the-fly)
SOURCE="data/raw/allMeSH_2021.jsonl"

# If you have already preprocessed the data, you will have a folder. Use the folder instead.
# SOURCE="output_folder_from_preprocessing"
# In that case, `test-size`, `train-years` and `test-years` will be taken from the preprocessed folder

grants-tagger train bertmesh \
    "" \
    $SOURCE \
    --test-size 10000 \
    --output_dir bertmesh_outs/pipeline_test/ \
    --train-years 2016,2017,2018,2019 \
    --test-years 2020,2021 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --multilabel_attention True \
    --freeze_backbone False \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --dropout 0.1 \
    --hidden_size 1024 \
    --warmup_steps 1000 \
    --fp16 \
    --torch_compile \
    --evaluation_strategy epoch \
    --eval_accumulation_steps 20 \
    --save_strategy epoch \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY}
