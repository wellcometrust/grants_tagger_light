# Run on p2.8xlarge instance
grants-tagger train bertmesh \
    "" \
    data/raw/allMeSH_2021.jsonl \
    --test-size 0.005 \
    --shards 250 \
    --output_dir bertmesh_outs/pipeline_test/ \
    --per_device_train_batch_size 32 \
    --num_train_epochs 1 \
    --fp16 \
    --torch_compile \
    --evaluation_strategy no \
    --save_strategy no \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY}
    #--save_strategy steps 
    #--save_steps 50000
    #--per_device_eval_batch_size 8 \
    #--eval_steps 50000 \
    #--evaluation_strategy steps
    #--report_to none
