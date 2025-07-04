import itertools
import json
import copy
import os
import numpy as np

def generate_combinations(base_params, repeat_times=1):
    """
    Generate all possible parameter combinations from a dictionary of parameter lists and repeat each combination.
    
    :param base_params: A dictionary where the values are lists of possible values for each parameter.
    :param repeat_times: Number of times each experiment configuration should be repeated.
    :return: A list of dictionaries, each representing one combination of parameters.
    """
    # Extract parameter keys and value lists
    keys = base_params.keys()
    value_lists = base_params.values()
    
    # Generate all combinations of parameter values
    combinations = list(itertools.product(*value_lists))
    
    # Convert combinations into a list of dictionaries and repeat them
    experiments = []
    for combination in combinations:
        experiment_params = dict(zip(keys, combination))
        for _ in range(repeat_times):  # Repeat each combination 'repeat_times' times
            experiments.append(copy.deepcopy(experiment_params))
    
    return experiments

def generate_log_interval_list(start, end, num):
    """
    生成在指定区间内的对数间隔列表。
    
    参数:
    start (int or float): 区间开始值。
    end (int or float): 区间结束值。
    num (int): 列表中值的数量。
    
    返回:
    list: 对数间隔的列表，包含指定数量的值。
    """
    return np.logspace(np.log10(start), np.log10(end), num=num).tolist()

def generate_lin_interval_list(start, end, num):
    """
    生成在指定区间内的线性间隔列表。

    参数:
    start (int or float): 区间开始值。
    end (int or float): 区间结束值。
    num (int): 列表中值的数量。

    返回:
    list: 线性间隔的列表，包含指定数量的值。
    """
    return np.linspace(start, end, num=num).tolist()
# /data/home/yekai/github/DiffAdvPerturbationBench/datasets/auxdatas/gen_aux_exps/SDS_SD21_VGGFace2_r8_id0
# Define the possible values for each parameter
# test_lable = "Orimetacloak4_total480_r6_idx50"
# test_lable = "Mist_SD21_VGGFace2_random50_mytarget2e-1_dynamic_r12"
test_lable = "ASPL_SD21_VGGFace2_r8_idx10_1id1atk"

params_options = {
    "DPAB_path":["/data/home/yekai/github/DiffAdvPerturbationBench"],
    # VGGFace2-clean/wikiart-data
    "dataset_name":["VGGFace2-10x64"],
    # "exp_batch_path": ["Algorithms/Diff-Protect/exp_datas_output/SDS_SD21_VGGFace2_random10_r8p16p12p4"],
    "exp_batch_path": ["datasets/auxdatas/gen_aux_exps/ASPL_SD21_VGGFace2_r8_idx10_1id1atk"],
    "pretrained_model_name_or_path": ["/data/home/yekai/github/DiffAdvPerturbationBench/SD/stable-diffusion-2-1-base"],
    "wandb_project_name": ["Dreambooth_train"],
    "mixed_precision": ["fp16"],
    "dreambooth_training_steps": [2000],
    "db_lr":[5e-7],
    # "a photo of sks person"
    "instance_prompt": ["a photo of sks person"],
    # "instance_prompt": ["a painting of sks artwork"],
    # "a photo of sks person;a dslr photo of sks person"
    # "inference_prompts": ["a painting of sks artwork"],
    "inference_prompts": ["a photo of sks person"],
    # painting/person
    # "class_name": ["painting"],
    "class_name": ["person"],
    "eval_gen_img_num": [16],
    # aspl50 fsmg100 SimAC50 mist100 sds100
    "use_sample_steps": ["50"],
}

# Number of times to repeat each configuration
# do not change
repeat_times = 1


for key, value in params_options.items():
    use_log = True
    if type(value) is list:
        continue
    if value.startswith("lin:") or value.startswith("log:"):
        if value.startswith("lin:"):
            use_log = False
        args = value.split(":")[1:]
        assert len(args) == 3, "Invalid format for linear/log interval list"
        start, end, num = map(float, args)
        num = int(num)
        if use_log:
            params_options[key] = generate_log_interval_list(start, end, num)
        else:
            params_options[key] = generate_lin_interval_list(start, end, num)

# Generate all combinations
experiments = generate_combinations(params_options, repeat_times=repeat_times)

# Wrap in "untest_args_list" for proper JSON structure
output = {"test_lable":test_lable ,"settings":params_options,"untest_args_list": experiments}

# Print the generated combinations as JSON
print(json.dumps(output, indent=4))

py_path = os.path.dirname(os.path.abspath(__file__))
# Save to a JSON file
with open(f"{py_path}/{test_lable}.json", "w") as outfile:
    json.dump(output, outfile, indent=4)
