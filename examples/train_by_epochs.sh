# Run on g5.12xlargeinstance

# In this case, `test-size`, `train-years` and `test-years` will be taken from the preprocessed folder
SOURCE="output_folder_from_preprocessing"


grants-tagger train bertmesh \
    "" \
    $SOURCE \
    --output_dir bertmesh_outs/pipeline_test/ \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --multilabel_attention True \
    --freeze_backbone unfreeze_bias \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --dropout 0.1 \
    --hidden_size 1024 \
    --warmup_steps 1000 \
    --max_grad_norm 5.0 \
    --scheduler-type cosine \
    --threshold 0.25 \
    --fp16 \
    --torch_compile \
    --evaluation_strategy epoch \
    --eval_accumulation_steps 20 \
    --save_strategy epoch \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY}
