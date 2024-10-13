export ACCELERATE_CONFIG_FILE="/home/featurize/.cache/huggingface/accelerate/default_config.yaml"
export PROJECT_DIR="/home/featurize/SemVarBench/code/dpo_finetune/"
export DATA_DIR="/home/featurize/trainingset_all_infos/"
cd ${PROJECT_DIR}

model_name="stabilityai/stable-diffusion-xl-base-1.0"
vae_name="madebyollin/sdxl-vae-fp16-fix"
train_data_dir=${DATA_DIR}"images/"
train_prompt_dir=${DATA_DIR}"prompts/"
output_dir=${PROJECT_DIR}"data/output/diffusion-sdxl-dpo/"
reward_match_root=${DATA_DIR}"reward/match/gpt-4v/scores/sd-xl-1-0/"
reward_mismatch_root=${DATA_DIR}"reward/mismatch/gpt-4v/scores/sd-xl-1-0/"
script=${PROJECT_DIR}"src/train_diffusion_dpo_sdxl_unet_text.py"
high_threshold=80
zero_threshold=10
threshold_filter="all"

accelerate launch --config_file $ACCELERATE_CONFIG_FILE --dynamo_backend=no "${script}" \
  --pretrained_model_name_or_path=${model_name}  \
  --pretrained_vae_model_name_or_path=${vae_name} \
  --output_dir="${output_dir}" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=4 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --checkpointing_steps=200 \
  --train_text_encoder \
  --train_data_dir=${train_data_dir} \
  --train_prompt_dir=${train_prompt_dir} \
  --reward_match_root=${reward_match_root} \
  --reward_mismatch_root=${reward_mismatch_root} \
  --high_threshold=${high_threshold} \
  --zero_threshold=${zero_threshold} \
  --threshold_filter=${threshold_filter} 
  # --report_to="wandb" 
  # --train_unet
  # --report_to="wandb" 
  # --run_validation --validation_steps=20 \
  # --seed="0" \
  # --cache_dir=${cache_dir}
  # --report_to="wandb" \
  # --push_to_hub

