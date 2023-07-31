# Run on p2.8xlarge instance
grants-tagger train bertmesh \
    "" \
    kk \
    --test-size 0.0025 \
    --shards 48 \
    --output_dir bertmesh_outs/pipeline_test/ \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --fp16 \
    --torch_compile \
    --evaluation_strategy steps \
    --save_strategy steps \
    --eval_steps 50000 \
    --save_steps 50000 \
    --wandb_project wellcome-mesh \
    --wandb_name test-train-all \
    --wandb_api_key ${WANDB_API_KEY} \
    --eval_accumulation_steps 20
    #--save_strategy steps 
    #--save_steps 50000
    #--per_device_eval_batch_size 8 \
    #--eval_steps 50000 \
    #--evaluation_strategy steps
    #--report_to none
