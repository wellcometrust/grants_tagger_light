# Run on g5.12xlarge instance

# After preprocessing
SOURCE="[SET_YOUR_PREPROCESSING_FOLDER_HERE]"

grants-tagger train bertmesh \
    "" \
    $SOURCE \
    --output_dir bertmesh_outs/pipeline_test/ \
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
