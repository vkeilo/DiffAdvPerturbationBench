#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
export CUDA_VISIBLE_DEVICES=2
# export task_name="Mist_SD21_VGGFace2_random50_mytarget2e-1_dynamic_r12"
export task_name="SimAC_SD21_VGGFace2_random50_r8_aux13_atk1"

python3 train_dreambooth.py \
    --pretrained_model_name_or_path="/data/home/yekai/github/DiffAdvPerturbationBench/SD/stable-diffusion-2-1-base" \
    --output_dir="/data/home/yekai/github/DiffAdvPerturbationBench/multi-dreambooth/train_outputs/${task_name}" \
    --revision="fp16" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --seed=1337 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_text_encoder \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --sample_batch_size=4 \
    --max_train_steps=10000 \
    --save_interval=10000 \
    --save_sample_prompt="a photo of qem person" \
    --concepts_list="concepts_lists/${task_name}.json"
