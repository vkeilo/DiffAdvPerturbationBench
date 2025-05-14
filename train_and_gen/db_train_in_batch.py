import os
import json
import time
import re
import argparse
import numpy as np

train_output_dir_name = "train_output"

def set_env_with_expansion(key, value):
    """
    将环境变量值中的 ${VAR_NAME} 替换为对应的环境变量值，并设置新的环境变量。
    
    :param key: 要设置的环境变量名称
    :param value: 包含 ${VAR_NAME} 的字符串，表示需要解析的环境变量值
    """
    # 使用正则表达式找到 ${VAR_NAME} 并替换为对应的环境变量值
    pattern = re.compile(r'\$\{([^}]+)\}')
    expanded_value = pattern.sub(lambda match: os.getenv(match.group(1), match.group(0)), value)
    
    # 设置环境变量
    os.environ[key] = expanded_value
    print(f"Environment variable '{key}' set to: {expanded_value}")

def test_one_args(args,test_lable):
    for k,v in args.items():
        if "$" in str(v):
            set_env_with_expansion(k,v)
        else:
            os.environ[k] = str(v)

    
    # os.chdir("..")
    # bash run : nohup bash script/gen_and_eval_vk.sh > output_MAT-1000-200-6-6-x1x1-radius11-allSGLD-rubust0.log 2>&1
    # os.environ["test_timestamp"] = str(int(time.time()))
    # output_path = os.path.join(os.getenv("DPAB_path"),'exp_datas_output_antidrm',os.getenv("exp_batch_name"))
    # logs_path = os.path.join(os.getenv("DPAB_path"),'logs_output_antidrm',os.getenv("exp_batch_name"))
    exp_batch_name = os.path.basename(os.getenv("exp_batch_path"))
    exp_batch_abs_path = os.path.abspath(os.path.join(os.getenv("DPAB_path"),os.getenv("exp_batch_path")))
    logs_path = os.path.join(os.getenv("DPAB_path"),'db_train_logs',exp_batch_name)
    output_path = exp_batch_abs_path
    if not os.path.exists(output_path):
        exit(f"target_path {output_path} not exist")
    if not os.path.exists(logs_path):
        print(f"logs_path {logs_path} not exist, create it")
        os.makedirs(logs_path)
    # only list path , not file
    run_name_list = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]
    print(f'run_name_list: {run_name_list}')
    finished_run_name_list = os.listdir(logs_path)
    for run_name in run_name_list:
        exp_abs_path = os.path.join(exp_batch_abs_path,run_name)
        print(os.listdir(exp_abs_path))
        tmp_log_path = os.path.join(logs_path, f"{run_name}.log")
        if f'{run_name}.log' in finished_run_name_list and check_file_end(tmp_log_path, "train and gen finished") and (train_output_dir_name in os.listdir(exp_abs_path)):
            print(f"run {run_name} finished, skip it") 
            continue
        print(f'processing run: {run_name}')
        os.environ['exp_run_name'] = run_name
        os.environ['wandb_run_name'] = run_name
        os.system(f"nohup bash train_and_gen/db_train_and_gen.sh > {tmp_log_path} 2>&1")
        check_file_for_pattern(f"{tmp_log_path}","train and gen finished")
    return exp_batch_name

def update_finished_json(finished_log_json_path, run_name):
    if not os.path.exists(finished_log_json_path):
        with open(finished_log_json_path, "w") as f:
            json.dump({}, f)
    finished_file = json.load(open(finished_log_json_path))
    # if json is empty, add key finished_args_list and value []
    if "finished_args_list" not in finished_file:
        finished_file["finished_args_list"] = []
    finished_file["finished_args_list"].append(run_name)
    json.dump(finished_file, open(finished_log_json_path, "w"))

def update_untest_json(untest_args_json_path):
    json_dict = json.load(open(untest_args_json_path))
    json_dict["untest_args_list"].pop(0)
    json.dump(json_dict, open(untest_args_json_path, "w"))

def check_file_for_pattern(file_path, pattern="find function last"):
    while True:
        try:
            # 打开文件并读取最后一行
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()  # 获取最后一行并去除两边空白
                    print(f"检测到的最后一行: {last_line}")
                    # 检查最后一行是否以指定模式开头
                    if last_line.startswith(pattern):
                        print("找到匹配的行，退出检测。")
                        return last_line
        except Exception as e:
            print(f"读取文件时出错: {e}")
        
        # 等待 3 分钟（180 秒）
        print("未找到匹配的行，等待 3 分钟后重新检测...")
        time.sleep(180)

def check_file_end(file_path, pattern="train and gen finished"):
    try:
        # 打开文件并读取最后一行
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()  # 获取最后一行并去除两边空白
                # 检查最后一行是否以指定模式开头
                if last_line.startswith(pattern):
                    print("找到匹配的行，退出检测。")
                    return True
    except Exception as e:
        print(f"读取文件时出错: {e}")
    print(f"not complete logfile, rerun~")
    return False

if __name__ == "__main__":
    print("batch test start...")
    # run in dir MetaCloak
    ADB_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    Pro_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.environ["ADB_PROJECT_ROOT"] = ADB_path
    os.environ["PYTHONPATH"] = str(os.getenv("PYTHONPATH")) + ":" + ADB_path + ":" + Pro_path
    
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="untest.json")
    parser.add_argument("--save_path", type=str, default="finished.json")
    parser.add_argument("--device_n", type=str, default="0")

    os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device_n
    untest_args_json_path = "train_and_gen/"+ parser.parse_args().target
    finished_log_json_path = "train_and_gen/" + parser.parse_args().save_path
    untest_file_con = json.load(open(untest_args_json_path))
    untest_args_list = untest_file_con["untest_args_list"].copy()
    test_lable = untest_file_con["test_lable"]
    print(f"test_lable: {test_lable}")
    for args in untest_args_list:
        print(f"start run :{args}")
        finished_name = test_one_args(args,test_lable)
        print(f"finished run :{finished_name}")
        update_untest_json(untest_args_json_path)
        update_finished_json(finished_log_json_path, finished_name)
    if not os.path.exists("finished_test"):
        os.mkdir("finished_test")
    os.system(f"mv {untest_args_json_path} finished_test")



    