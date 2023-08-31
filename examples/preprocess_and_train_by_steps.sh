# Run on g5.12xlargeinstance

# Without preprocessing (on-the-fly)
SOURCE="data/raw/allMeSH_2021.jsonl"

grants-tagger train bertmesh \
    "" \
    $SOURCE \
    --output_dir bertmesh_outs/pipeline_test/ \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --multilabel_attention True \
    --freeze_backbone False \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --dropout 0.1 \
    --hidden_size 1024 \
    --warmup_steps 1000 \
    --max_grad_norm 5.0 \
    --scheduler-type cosine \
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
