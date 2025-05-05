import argparse
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import re
# from eval_score import get_score
import numpy as np
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from data_utils import PromptDataset, load_data_by_picname

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
    print(f'scores: {scores}')
    # mean_scores = torch.mean(scores)
    # 100
    return torch.mean(scores)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="Image with Adversarial Perturbation.",
    )
    parser.add_argument(
        "--clean_img_dir",
        type=str,
        default=None,
        required=True,
        help="Clean imgs of protected instance",
    )
    parser.add_argument(
        "--clean_ref_db",
        type=str,
        default=None,
        required=True,
        help="Ref imgs of protected instance",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--gen_img_dir",
        type=str,
        default=None,
        required=False,
        help="Customized and fine tuned generated images",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )

    args = parser.parse_args()
    return args

# exp_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori"

# ori_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/image_before_addding_noise"
# noisy_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/noise-ckpt/final"
# gen_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/train_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7-gau-gau-eval/gen-release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7-dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks-eval-gau-rate-/0_DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person"

def main(args):
    instance_data_dir = args.instance_data_dir
    clean_img_dir = args.instance_data_dir
    clean_ref_dir = args.clean_ref_dir
    gen_img_dir = args.gen_img_dir
    class_data_dir = args.class_data_dir
    instance_prompt = args.instance_prompt
    class_prompt = args.class_prompt
    
    target_path = "/data/home/yekai/github/MetaCloak/exp_datas_output_antidrm/Orimetacloak4_total480_r6_idx50"
    score_dict = {"max_noise_r":[],"noise_L0":[],"pix_change_mean":[],"change_area_mean":[],"ciede2000_score":[]}

    def extract_id(s):
        match = re.search(r'-id(\d+)-', s)
        return int(match.group(1)) if match else float('inf')

    dir_list= os.listdir(target_path)
    sorted_dirlist = sorted(dir_list, key=extract_id)

    for exp_dir in sorted_dirlist:
        print(exp_dir)
        exp_pahth = os.path.join(target_path,exp_dir)
        clean_ref_dir = os.path.join(exp_pahth,'image_clean')
        ori_pics_dir = os.path.join(exp_pahth,"image_before_addding_noise")
        noisy_pics_dir = os.path.join(exp_pahth,f"noise-ckpt/{rounds}")
        # gen_pics_dir = os.path.join(exp_pahth,"train_output/dreambooth/a_photo_of_sks_person")
        # clean_ref_dir = ori_pics_dir

        # score_dict = get_score(gen_pics_dir,clean_ref_dir)
        # print(score_dict)
        # k_list = list(score_dict.keys())
        # for k in k_list:
        #     means = []
        #     means.append(score_dict[k])
        #     print(f"{k}_mean {np.mean(means)}")
        #     stds = []
        #     stds.append(score_dict[k])
        #     print(f"{k}_std {np.std(stds)}")



        original_data = load_data_by_picname(ori_pics_dir)
        perturbed_data = load_data_by_picname(noisy_pics_dir)
        max_noise_r = find_max_pixel_change(perturbed_data, original_data)
        noise_L0 = get_L0(perturbed_data, original_data)
        noise_L1 = get_L1(perturbed_data, original_data)
        noise_p = get_change_p(perturbed_data, original_data)
        ciede2000_score = get_ciede2000_diff(original_data, perturbed_data)
        score_dict['max_noise_r'].append(max_noise_r)
        score_dict['noise_L0'].append(noise_L0)
        score_dict['pix_change_mean'].append(noise_L1/(512*512)/2)
        score_dict['change_area_mean'].append(noise_p*100)
        score_dict['ciede2000_score'].append(ciede2000_score)

    data_len = len(score_dict['max_noise_r'])
    all_max_noise_r = sum(score_dict['max_noise_r'])/data_len
    all_noise_L0 = sum(score_dict['noise_L0'])/data_len
    all_pix_change_mean = sum(score_dict['pix_change_mean'])/data_len
    all_change_area_mean = sum(score_dict['change_area_mean'])/data_len
    all_ciede2000_score = sum(score_dict['ciede2000_score'])/data_len



    print(f"max_noise_r {all_max_noise_r:.2f}")
    print(f"noise_L0 {all_noise_L0:.2f}")
    print(f"pix_change_mean {all_pix_change_mean:.2f}")
    print(f"change_area_mean {all_change_area_mean:.2f}")
    print(f"ciede2000_score {all_ciede2000_score:.6f}")
    print(len(score_dict['ciede2000_score']))
    for i in [t.item() for t in score_dict['ciede2000_score']]:
        print(f"{i:3f}")

if __name__ == "__main__":
    args = parse_args()

    main(args)
    print("exp finished")