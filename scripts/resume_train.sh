# Run on p2.8xlarge instance
grants-tagger train bertmesh \
    bertmesh_outs/pipeline_test/checkpoint-100000 \
    kk \
    --output_dir bertmesh_outs/pipeline_test_from_100000/ \
    --ignore_data_skip=True \
    --shards 48 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 2 \
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
