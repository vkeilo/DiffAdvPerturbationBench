# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# echo "script dir is $SCRIPT_DIR"
# export DPAB_path="$(dirname "$SCRIPT_DIR")"
export exp_batch_abs_path="$DPAB_path/$exp_batch_path"
export wandb_entity_name=vkeilo
export seed=0


export max_train_steps=$dreambooth_training_steps
echo set up the model path
echo $eval_model_path
export WANDB_MODE=offline
# export exp_path="$exp_batch_abs_path/$exp_run_name/"
export CLEAN_REF="$exp_batch_abs_path/$exp_run_name/image_clean_ref/"
export INSTANCE_DIR="$exp_batch_abs_path/$exp_run_name/noise-ckpt/$use_sample_steps/"
export INSTANCE_DIR_CLEAN="$exp_batch_abs_path/$exp_run_name/image_before_addding_noise/"

class_name_fixed=$(echo $class_name | sed "s/ /-/g")
export CLASS_DIR="$DPAB_path/prior-data/$class_name_fixed"

export DREAMBOOTH_OUTPUT_DIR="$exp_batch_abs_path/$exp_run_name/train_output"
echo "dreambooth output to $DREAMBOOTH_OUTPUT_DIR"

cd $DPAB_path
# source activate $ADB_ENV_NAME;
# vkeilo add it
# this is to indicate that whether we have finished the training before 
# training_finish_indicator=$DREAMBOOTH_OUTPUT_DIR/finished.txt

# skip training if instance data not exist 
if [ ! -d "$INSTANCE_DIR" ]; then
  echo "instance data $INSTANCE_DIR not exist, skip training"
  exit 1
fi

echo "start dreambooth training"
# vkeilo del --train_text_encoder \
command="""python train_and_gen/train_dreambooth_VV.py --clean_img_dir $INSTANCE_DIR_CLEAN --clean_ref_db $CLEAN_REF --class_name '$class_name' \
--wandb_entity_name $wandb_entity_name \
--seed $seed \
--gradient_checkpointing \
--pretrained_model_name_or_path='$pretrained_model_name_or_path'  \
--instance_data_dir='$INSTANCE_DIR' \
--class_data_dir='$CLASS_DIR' \
--output_dir=$DREAMBOOTH_OUTPUT_DIR \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--instance_prompt='${instance_prompt}' \
--class_prompt='a photo of ${class_name}' \
--inference_prompts='${inference_prompts}' \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--learning_rate=$db_lr \
--lr_scheduler=constant \
--lr_warmup_steps=0 \
--num_class_images=200 \
--max_train_steps=$dreambooth_training_steps \
--center_crop \
--sample_batch_size=4 \
--use_8bit_adam \
--eval_gen_img_num=$eval_gen_img_num \
--wandb_project_name $wandb_project_name \
--poison_rate 1.0 \
"""

if [ "$eval_model_name" = "SD21base" ] || [ "$eval_model_nam" = "SD21" ]; then
  command="$command --enable_xformers_memory_efficient_attention --mixed_precision=bf16 --prior_generation_precision=bf16"
else 
  command="$command --mixed_precision=bf16 --prior_generation_precision=bf16"
fi 

# # check variable more_defense_name 
# if [ -z "$more_defense_name" ]; then
#   more_defense_name="none"
# fi

# # if more_defense_name is not none, then add the flag
# if [ "$more_defense_name" != "none" ]; then
#   # if more_defense_name is sr, then add the flag
#   if [ "$more_defense_name" = "sr" ]; then
#     command="$command --transform_sr"
#   fi
#   # transform_tvm
#   if [ "$more_defense_name" = "tvm" ]; then
#     command="$command --transform_tvm"
#   fi
#   # jpeg_transform
#   if [ "$more_defense_name" = "jpeg" ]; then
#     command="$command --jpeg_transform"
#   fi
# fi

if [ "$eval_mode" = "gau" ]; then
  command="$command --transform_defense --transform_gau --gau_kernel_size $gauK --transform_hflip"
fi
echo $command
eval $command


mv $DREAMBOOTH_OUTPUT_DIR/checkpoint-$dreambooth_training_steps/dreambooth $DREAMBOOTH_OUTPUT_DIR
rm -r $DREAMBOOTH_OUTPUT_DIR/checkpoint-$dreambooth_training_steps