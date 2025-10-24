
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import glob
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import torch
import re
from eval_score import get_score
import numpy as np
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from data_utils import PromptDataset, load_data_by_picname
import pandas as pd
from tqdm import tqdm
import argparse
import gc
from torch_fidelity import calculate_metrics

def find_max_pixel_change(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    
    # Find the maximum pixel difference
    max_change = torch.max(diff)
    
    return max_change.item()

def get_L0(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    diff_L0 = torch.sum(diff > 0, dim=(1, 2, 3))
    # Find the maximum pixel difference
    mean_L0 = torch.mean(diff_L0.float())
    return mean_L0.item()

def get_L1(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    diff_L1 = torch.sum(diff, dim=(1, 2, 3))
    mean_L1 = torch.mean(diff_L1.float())
    return mean_L1.item()

def get_change_p(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    diff_L0_all = torch.sum(diff > 0, dim=(0, 1, 2, 3))
    pix_num_all = original_img.shape[0] * original_img.shape[1] * original_img.shape[2] * original_img.shape[3]
    change_p = diff_L0_all / pix_num_all
    return change_p.item()

def get_ciede2000_diff(ori_imgs,advimgs):
    device = torch.device('cuda')
    ori_imgs_0_1 = ori_imgs/255
    advimgs_0_1 = advimgs/255
    advimgs_0_1.clamp_(0,1)
    # print(f'ori_imgs_0_1.min:{ori_imgs_0_1.min()}, ori_imgs_0_1.max:{ori_imgs_0_1.max()}')
    # print(f'advimgs_0_1.min:{advimgs_0_1.min()}, advimgs_0_1.max:{advimgs_0_1.max()}')
    X_ori_LAB = rgb2lab_diff(ori_imgs_0_1,device)
    advimgs_LAB = rgb2lab_diff(advimgs_0_1,device)
    # print(f'advimgs: {advimgs}')
    # print(f'ori_imgs: {ori_imgs}')
    color_distance_map=ciede2000_diff(X_ori_LAB,advimgs_LAB,device)
    # print(color_distance_map)
    scores = torch.norm(color_distance_map.view(ori_imgs.shape[0],-1),dim=1)
    # print(f'scores: {scores}')
    # mean_scores = torch.mean(scores)
    # 100
    return torch.mean(scores)

def move_column_to_front(df, column_name):
    cols = [column_name] + [c for c in df.columns if c != column_name]
    return df[cols]

# exp_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori"

# ori_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/image_before_addding_noise"
# noisy_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/noise-ckpt/final"
# gen_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/train_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7-gau-gau-eval/gen-release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7-dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks-eval-gau-rate-/0_DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="None")
    parser.add_argument("--round", type=str, default="0")
    parser.add_argument("--device_n", type=str, default="0")
    parser.add_argument("--type", type=str, default="face")
    parser.add_argument("--prompt", type=str, default="a_photo_of_sks_person")

    os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device_n
    # target_path = "/data/home/yekai/github/DiffAdvPerturbationBench/Algorithms/Anti-DreamBooth/exp_datas_output/aspl_VGGFace2_random50_r4p8p12p16"
    target_path = parser.parse_args().target
    df_save_path = f"{script_dir}/eval_result"
    # rounds = "50"
    rounds = parser.parse_args().round


    # gen_prompt = "a_photo_of_sks_person"
    # gen_prompt = "a_painting_of_sks_artwork"
    gen_prompt = parser.parse_args().prompt

    device = torch.device("cuda")
    # score_dict = {"max_noise_r":[],"noise_L0":[],"pix_change_mean":[],"change_area_mean":[],"ciede2000_score":[]}
    # eval_items = ["exp_run_name","max_noise_r","noise_L0","pix_change_mean","change_area_mean","ciede2000_score",'SDS','CLIP_Face_IQA','LIQE_Scene_Human','CLIPIQA','BRISQUE','LIQE_Quality','IMS_CLIP_ViT-B/32','CLIP_IQAC','IMS_VGG-Face_cosine','PSNR_ref','PSNR_train']
    eval_items = ["exp_run_name",
                    "max_noise_r",
                    "noise_L0",
                    "pix_change_mean",
                    "change_area_mean",
                    "ciede2000_score",
                    'SDS',
                    'CLIP_Face_IQA',
                    'LIQE_Scene_Human',
                    'CLIPIQA',
                    'BRISQUE',
                    'LIQE_Quality',
                    'IMS_CLIP_ViT-B/32',
                    'CLIP_IQAC',
                    'IMS_VGG-Face_cosine',
                    'PSNR_ref',
                    'PSNR_train',
                    'FID_ref',
                    'FID_train',
                    'LPIPS',
                    'SSIM',
                    'PSNR_perturbation',
                    # 'CLIP_ISIMC'
                    # 'precision',
                    # 'recall'
                ]
    print(f"now eval {eval_items}")

    exp_batch_name = target_path.split('/')[-1]
    def extract_id(s):
        match = re.search(r'-id(\d+)-', s)
        return int(match.group(1)) if match else float('inf')

    dir_list= os.listdir(target_path)
    sorted_dirlist = sorted(dir_list, key=extract_id)

    eval_data = pd.DataFrame(columns=eval_items)
    for exp_dir in tqdm(sorted_dirlist):
        score_dict = {}
        score_dict["exp_run_name"] = exp_dir
        print(exp_dir)
        exp_path = os.path.join(target_path,exp_dir)
        clean_ref_dir = os.path.join(exp_path,'image_clean_ref')
        ori_pics_dir = os.path.join(exp_path,"image_before_addding_noise")
        noisy_pics_dir = os.path.join(exp_path,f"noise-ckpt/{rounds}")
        if rounds == "*":
            matched = glob.glob(os.path.join(exp_path, "noise-ckpt", rounds))
            if len(matched) != 1:
                raise ValueError(f"Expected exactly one match, but found {len(matched)}: {matched}")
            noisy_pics_dir = os.path.abspath(matched[0])
        gen_pics_dir = os.path.join(exp_path,f"train_output/dreambooth/{gen_prompt}")
        print(gen_pics_dir)
        score_from_tool = get_score(gen_pics_dir,clean_ref_dir,clean_img_dir=ori_pics_dir, perturbed_img_dir=noisy_pics_dir,eval_items=eval_items,type_name=parser.parse_args().type)
        # score_dict.update(score_from_tool)
        # print(score_dict)
        k_list = list(score_from_tool.keys())
        for k in k_list:
            means = []
            means.append(score_from_tool[k])
            # print(f"{k}_mean {np.mean(means)}")
            # stds = []
            # stds.append(score_from_tool[k])
            # print(f"{k}_std {np.std(stds)}")
            score_dict[k] = np.mean(means)
        # print(("','".join(k_list)))
        del score_from_tool
        torch.cuda.empty_cache()
        gc.collect()

        # PR
        # metrics = calculate_metrics(
        #     input1=gen_pics_dir,   
        #     input2=clean_ref_dir,   
        #     cuda=True,                            
        #     isc=False, fid=False, kid=False,      
        #     pr=True                               
        # )
        # score_dict['precision'] = metrics['precision']
        # score_dict['recall'] = metrics['recall']

        original_data = load_data_by_picname(ori_pics_dir).to(device=device)
        perturbed_data = load_data_by_picname(noisy_pics_dir).to(device=device)
        max_noise_r = find_max_pixel_change(perturbed_data, original_data)
        noise_L0 = get_L0(perturbed_data, original_data)
        noise_L1 = get_L1(perturbed_data, original_data)
        noise_p = get_change_p(perturbed_data, original_data)
        ciede2000_score = get_ciede2000_diff(original_data, perturbed_data).to('cpu').detach().numpy()
        score_dict['max_noise_r'] = max_noise_r
        score_dict['noise_L0'] = noise_L0
        score_dict['pix_change_mean'] = noise_L1/(512*512)/2
        score_dict['change_area_mean'] = noise_p*100
        score_dict['ciede2000_score'] = ciede2000_score

        eval_data.loc[len(eval_data)] = score_dict

    print(eval_data)

    eval_data = move_column_to_front(eval_data,"exp_run_name")
    # eval_data.to_csv(f"{df_save_path}/{exp_batch_name}.csv", index=False)
    csv_path = f"{df_save_path}/{exp_batch_name}.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        # 如果文件不存在，直接保存当前数据
        eval_data.to_csv(csv_path, index=False)
    else:
        # 文件存在时，读取现有CSV数据
        old_df = pd.read_csv(csv_path)
        
        # 确定当前数据中存在但现有CSV中不存在的新列
        new_columns = eval_data.columns.difference(old_df.columns)
        
        if not new_columns.empty:
            # 提取当前数据中的exp_run_name和新列，避免引入旧列干扰
            # 使用reindex确保列顺序正确，仅保留需要的列
            # eval_new_data = eval_data.reindex(columns=['exp_run_name'] + new_columns.tolist())
            
            # # 合并数据：以exp_run_name为键，外连接保留所有实验名称
            # # 这样会合并新旧数据，新列在现有数据中无值则为NaN
            # merged_df = old_df.merge(eval_new_data, on='exp_run_name', how='outer')
            
            # # 保存合并后的数据（覆盖原文件）
            # print(f"{csv_path} already exsists, now add columns:{new_columns}")
            # merged_df.to_csv(csv_path, index=False)

            eval_new_data = eval_data[["exp_run_name"] + list(new_columns)]        
            # 以旧数据为基准，左连接合并新列
            # 保持旧数据行顺序，且仅追加新列
            merged_df = old_df.merge(
                eval_new_data, 
                on="exp_run_name", 
                how="left"  # 左连接确保行顺序不变
            )
            
            # 覆盖保存，保持原有顺序和列顺序
            print(f"{csv_path} already exsists, now add columns:{new_columns}")
            merged_df.to_csv(csv_path, index=False)
        else:
            # 如果没有新增列，可以选择不执行任何操作或根据需求处理
            # 例如，若需更新现有列数据，可在此处添加逻辑
            pass
