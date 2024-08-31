export HF_DATASETS_CACHE="/workspace/compe/.cache"
NUM_GPU=1
GPU_IDS="0"
model_name_or_path=x2bee/POLAR-14B-v0.5
rank=32
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python train.py \
    --seed 42 \
    --output_dir "models/${model_name_or_path}" \
    --model_name_or_path ${model_name_or_path} \
    --model_max_length 2048 \
    --train_data_path "data/train.json" \
    --valid_data_path "data/dev.json" \
    --aug_data_path "data/aug_train.json" \
    --do_aug True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "steps" \
    --eval_strategy "steps" \
    --eval_steps "0.16" \
    --save_steps "0.16" \
    --logging_strategy "steps" \
    --logging_steps "1" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --dataloader_num_workers "4" \
    --remove_unused_columns "True" \
    --r ${rank} \
    --lora_alpha ${rank} \
    --lora_dropout 0.1 \
    --target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --use_rslora "True" \
    --metric_for_best_model "accuracy" \
    --load_best_model_at_end \
    --greater_is_better "True" \
    --neftune_noise_alpha 15.0 \
    --gradient_checkpointing True \
    --bf16