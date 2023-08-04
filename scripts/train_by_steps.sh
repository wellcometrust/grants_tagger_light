# Run on g5.12xlargeinstance

# Without preprocessing (on-the-fly)
SOURCE="data/raw/allMeSH_2021.jsonl"

# After preprocessing first
# SOURCE="output_folder_from_preprocessing"

grants-tagger train bertmesh \
    "" \
    $SOURCE \
    --test-size 10000 \
    --output_dir bertmesh_outs/pipeline_test/ \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --dropout 1.0 \
    --warmup_steps 1000 \
    --fp16 \
    --torch_compile \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --eval_accumulation_steps 20 \
    --save_strategy steps \
    --save_steps 50000 \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY}
