# Run on p2.8xlarge instance
grants-tagger train bertmesh \
    "" \
    data/raw/allMeSH_2021.json \
    -1 \
    --output_dir bertmesh_outs/pipeline_test/ \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --evaluation_steps 100000 \
    --fp16 \
    --torch_compile
