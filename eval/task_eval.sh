SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "script dir is $SCRIPT_DIR"
export DPAB_path="$(dirname "$SCRIPT_DIR")"
export exp_batch_abs_path="$DPAB_path/$exp_batch_path"
export wandb_entity_name=vkeilo
export seed=0


export WANDB_MODE=offline
export CLEAN_REF="$exp_batch_abs_path/$exp_run_name/image_clean_ref/"
export INSTANCE_DIR="$exp_batch_abs_path/$exp_run_name/noise-ckpt/$use_sample_steps/"
export INSTANCE_DIR_CLEAN="$exp_batch_abs_path/$exp_run_name/image_before_addding_noise/"
export DREAMBOOTH_OUTPUT_DIR="$exp_batch_abs_path/$exp_run_name/train_output"
export EVAL_OUTPUT_DIR="$exp_batch_abs_path/$exp_run_name/eval_output"
echo "dreambooth output to $DREAMBOOTH_OUTPUT_DIR"

# class_name_fixed=$(echo $class_name | sed "s/ /-/g")
# export CLASS_DIR="$DPAB_path/prior-data/$class_name_fixed"


cd $DPAB_path
# source activate $ADB_ENV_NAME;
# vkeilo add it
# this is to indicate that whether we have finished the training before 
# training_finish_indicator=$DREAMBOOTH_OUTPUT_DIR/finished.txt

# skip training if instance data not exist 
if [ ! -d "$DREAMBOOTH_OUTPUT_DIR" ]; then
  echo "生成的图片 $DREAMBOOTH_OUTPUT_DIR not exist, skip evaling"
  exit 1
fi

echo "start evaling"

command="""python eval/eval_genpics.py \
--clean_img_dir $INSTANCE_DIR_CLEAN \
--clean_ref_db $CLEAN_REF \
--class_name '$class_name' \
--instance_data_dir='$INSTANCE_DIR' \
--gen_img_dir='$DREAMBOOTH_OUTPUT_DIR' \
--class_data_dir='$CLASS_DIR' \
--output_dir=$EVAL_OUTPUT_DIR \
--instance_prompt='${instance_prompt}' \
--class_prompt='a photo of ${class_name}' \
"""


echo $command
eval $command
