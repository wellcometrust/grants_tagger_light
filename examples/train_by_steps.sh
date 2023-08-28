# Run on g5.12xlargeinstance

# If you have already preprocessed the data, you will have a folder. Use the folder instead.
SOURCE="output_folder_from_preprocessing"
# In that case, `test-size`, `train-years` and `test-years` will be taken from the preprocessed folder

grants-tagger train bertmesh \
    "" \
    $SOURCE \
    --output_dir bertmesh_outs/pipeline_test/ \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --multilabel_attention True \
    --freeze_backbone unfreeze \
    --num_train_epochs 5 \
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
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --eval_accumulation_steps 20 \
    --save_strategy steps \
    --save_steps 10000 \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY}
