# Run on g5.12xlarge instance

# Without saving (on-the-fly)
#SOURCE="data/raw/allMeSH_2021.jsonl"
SOURCE="data/raw/retagging/allMeSH_2021.2016-2021.jsonl"

grants-tagger train bertmesh \
    "" \
    $SOURCE \
    --test-size 25000 \
    --train-years 2016,2017,2018,2019 \
    --test-years 2020,2021 \
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
    --evaluation_strategy epochs \
    --eval_accumulation_steps 20 \
    --save_strategy epochs \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY}
