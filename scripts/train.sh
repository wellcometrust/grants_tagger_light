# Run on p2.8xlarge instance
CUDA_VISIBLE_DEVICES=0 grants-tagger train bertmesh \
    "" \
    kk/1.json \
    --report_to None
    --test-size 0.05 \
    --shards 250 \
    --output_dir bertmesh_outs/pipeline_test/ \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY} \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 100000 \
    --save_strategy steps \
    --save_steps 100000
    --ddp_backend gloo
    --fp16 \
    --torch_compile

