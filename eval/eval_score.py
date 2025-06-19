# -----------------------------------------------------------------------
# Copyright (c) 2023 Yixin Liu Lehigh University
# All rights reserved.
#
# This file is part of the MetaCloak project. Please cite our paper if our codebase contribute to your project. 
# -----------------------------------------------------------------------

from typing import Any
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from LIQE.LIQE import LIQE
import tensorflow as tf
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.io import read_image
from piq import CLIPIQA, BRISQUELoss
clipiqa = CLIPIQA()
brisque = BRISQUELoss()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
ckpt = f'{script_dir}/LIQE/checkpoints/LIQE.pt'
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
Metacloak_path = f"{os.path.dirname(script_dir)}/Algorithms/MetaCloak"
sys.path.append(Metacloak_path)
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32")
lieq_model = LIQE(ckpt, device = 'cuda' if torch.cuda.is_available() else 'cpu')
from robust_facecloak.generic.modi_deepface import find_without_savepkl
from deepface import DeepFace
import glob
import os
from skimage.metrics import peak_signal_noise_ratio
import lpips
from skimage.metrics import structural_similarity as ssim
from openai import OpenAI
import base64
import time
# from pytorch_fid import fid_score 

def loop_to_get_overall_score(gen_image_dir, clean_ref_dir="", func_get_score_of_one_image=None, type_name="face"):
    files_db_gen = glob.glob(os.path.join(gen_image_dir, "*.png"))
    files_db_gen += glob.glob(os.path.join(gen_image_dir, "*.jpg"))
    scores = []
    assert len(files_db_gen) > 0
    for i in range(len(files_db_gen)):
        gen_i = files_db_gen[i]
        score = func_get_score_of_one_image(gen_i, clean_ref_dir, type_name=type_name)
        scores.append(score)
    # filter out nan and np.inf 
    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]
    scores = scores[~np.isinf(scores)]
    scores = scores[~np.isinf(-scores)]
    return np.mean(scores)

class ScoreEval():
    def __init__(self, func_get_score_of_one_image=lambda image_dir, clean_ref_dir, clean_image_dir, type_name="face", mode=None: 0):
        self.func_get_score_of_one_image = func_get_score_of_one_image
    
    def __loop_to_get_overall_score__(self, gen_image_dir, clean_ref_db=None, clean_image_dir=None, type_name="face", mode=None):
        files_db_gen = glob.glob(os.path.join(gen_image_dir, "*.png"))
        files_db_gen += glob.glob(os.path.join(gen_image_dir, "*.jpg"))
        scores = []
        assert len(files_db_gen) > 0
        for i in range(len(files_db_gen)):
            gen_i = files_db_gen[i]
            score = self.func_get_score_of_one_image(gen_i, clean_ref_db, clean_image_dir, type_name=type_name, mode=mode)
            scores.append(score)
        # filter out nan and np.inf 
        scores = np.array(scores)
        scores = scores[~np.isnan(scores)]
        scores = scores[~np.isinf(scores)]
        scores = scores[~np.isinf(-scores)]
        # return np.mean(scores)
        return scores
    
    def __call__(self, gen_image_dir, clean_ref_db=None, clean_image_dir = None, type_name="face", mode = None):
        return self.__loop_to_get_overall_score__(gen_image_dir, clean_ref_db, clean_image_dir, type_name=type_name, mode=mode)
    

def BRISQUE_get_score(gen_i, clean_ref_db=None, clean_image_dir=None, type_name="face", mode=None):
    from PIL import Image
    PIL_image = Image.open(gen_i).convert("RGB")
    from torchvision import transforms
    trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    with torch.no_grad():
        score = brisque(trans(PIL_image).unsqueeze(0)).item()
    return score

BRISQUE_Scorer = ScoreEval(func_get_score_of_one_image=BRISQUE_get_score)

def CLIPIQA_get_score(gen_i, clean_ref_db=None, clean_image_dir=None,type_name="face", mode=None):
    from PIL import Image
    PIL_image = Image.open(gen_i).convert("RGB")
    from torchvision import transforms
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    with torch.no_grad():
        score = clipiqa(trans(PIL_image).unsqueeze(0)).item()
    return score
CLIP_IQA_Scorer = ScoreEval(func_get_score_of_one_image=CLIPIQA_get_score)


def LIQE_get_quality_score(gen_i, clean_ref_db=None, clean_image_dir=None,type_name="face", mode=None):
    img = Image.open(gen_i).convert('RGB')
    from torchvision.transforms import ToTensor
    img = ToTensor()(img).unsqueeze(0)
    q1, s1, d1 = lieq_model(img)
    return q1.item()
LIQE_Quality_Scorer = ScoreEval(func_get_score_of_one_image=LIQE_get_quality_score)

def LIQE_get_scene_human_score(gen_i, clean_ref_db=None, clean_image_dir=None,type_name="face", mode=None):
    img = Image.open(gen_i).convert('RGB')
    from torchvision.transforms import ToTensor
    img = ToTensor()(img).unsqueeze(0)
    q1, s1, d1 = lieq_model(img)
    return 1 if s1 == "human" else 0
LIQE_Scene_Human_Scorer = ScoreEval(func_get_score_of_one_image=LIQE_get_scene_human_score)


def IMS_CLIP_get_score(gen_i, clean_ref_db, clean_image_dir=None,type_name="face", mode=None):
    import torch
    img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(img).unsqueeze(0).to('cuda')
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    ref_pkl_path = os.path.join(clean_ref_db, "ref_mean_clip_vit_32.pkl")
    ref_representation_mean = None
    if os.path.exists(ref_pkl_path):
        ref_representation_mean= torch.load(ref_pkl_path)
    else:
        ref_images = glob.glob(os.path.join(clean_ref_db, "*.png"))
        ref_images += glob.glob(os.path.join(clean_ref_db, "*.jpg"))
        ref_representation_mean = 0.0
        for ref_image in ref_images:
            ref_image = Image.open(ref_image).convert('RGB')
            ref_image = clip_preprocess(ref_image).unsqueeze(0).to('cuda')
            with torch.no_grad():
                ref_representation_mean += clip_model.encode_image(ref_image).cpu()
        ref_representation_mean /= len(ref_images)
        ref_representation_mean = ref_representation_mean / ref_representation_mean.norm(dim=-1, keepdim=True)
        torch.save(ref_representation_mean, ref_pkl_path)
    
    # calculate cosine similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    cosine_similarity = (ref_representation_mean.cpu().numpy() * image_features.cpu().numpy()).sum().mean()
    return cosine_similarity

IMS_CLIP_Scorer = ScoreEval(func_get_score_of_one_image=IMS_CLIP_get_score)

def CLIP_Face_get_score(gen_i, clean_ref_db=None, clean_image_dir=None,type_name="face", mode=None):
    import torch
    gen_img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
    text = clip.tokenize(["good face", 'bad face']).to('cuda')
    similarity_matrix = None
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity_matrix[0][0].item() - similarity_matrix[0][1].item()

CLIP_Face_Scorer = ScoreEval(func_get_score_of_one_image=CLIP_Face_get_score)

def CLIP_IQAC_get_score(gen_i, clean_ref_db=None, clean_image_dir=None,type_name="face", mode=None):
    import torch
    gen_img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
    text = clip.tokenize(["a good photo of " + type_name, "a bad photo of " + type_name]).to('cuda')
    similarity_matrix = None
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity_matrix[0][0].item() - similarity_matrix[0][1].item()

CLIP_IQAC_Scorer = ScoreEval(func_get_score_of_one_image=CLIP_IQAC_get_score)

def CLIP_ISIMC_get_score(gen_i, clean_ref_db=None, clean_image_dir=None,type_name="face", mode=None):
    gen_img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
    text_list = [clip.tokenize([f"a {type_name} with {feature}", f"a {type_name} without {feature}"]).to('cuda') for feature in mode]
    # print(f"len of text_list: {len(text_list)}")
    score_list = []
    for text in text_list:
        similarity_matrix = None
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        score_list.append(similarity_matrix[0][0].item() - similarity_matrix[0][1].item())
    
    # print(str([f"{feature}: {score}" for feature, score in zip(mode, score_list)]))
    # for i in range(len(score_list)):
        # print(f"{mode[i]}: {score_list[i]}")
    # print(f"average score: {np.mean(score_list)}")
    return np.mean(score_list)
CLIP_ISIMC_Scorer = ScoreEval(func_get_score_of_one_image=CLIP_ISIMC_get_score)

def CLIP_zero_short_classification_get_score(gen_i, clean_ref_db=None, clean_image_dir=None,type_name="face", mode=None):
    import torch
    gen_img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
    text = clip.tokenize(["a picture of " + type_name, "a picture of non-" + type_name]).to('cuda')
    similarity_matrix = None
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity_matrix[0][0].item() - similarity_matrix[0][1].item()

CLIP_zero_short_classification_Scorer = ScoreEval(func_get_score_of_one_image=CLIP_zero_short_classification_get_score)

def IMS_get_score(gen_i, clean_ref_db, clean_image_dir=None, type_name="face", mode=["cosine","VGG-Face"]):
    distance_metric = mode[0]
    model_name = mode[1]
    dfs = find_without_savepkl(img_path = gen_i, db_path = clean_ref_db, enforce_detection=False, distance_metric=distance_metric, model_name=model_name, )
    all_scores = dfs[0][f'{model_name}_{distance_metric}'].values
    import numpy as np
    all_scores = all_scores[~np.isnan(all_scores)]
    dis = 0
    if len(all_scores)==0:
        dis = 2
    else:
        dis = np.mean(all_scores)
    return 1-dis

IMS_Face_Scorer = ScoreEval(func_get_score_of_one_image=IMS_get_score)

def FDSR_get_score(gen_i, clean_ref_db=None, clean_image_dir=None, model='retinaface', type_name="face", mode=None):
    face_obj = DeepFace.extract_faces(img_path = gen_i, 
        target_size = (224,224),
        detector_backend = model,
        enforce_detection = False,
        )
    score = 0
    for i in range(len(face_obj)):
        score += face_obj[i]['confidence']
    return score / len(face_obj)

FDSR_Scorer = ScoreEval(func_get_score_of_one_image=FDSR_get_score)


def PSNR_get_score(gen_i, clean_ref_db=None, clean_img_dir=None, type_name="face", mode='ref'):
    """PSNR指标计算函数"""
    assert mode in ['ref','ori','perturbation']
    if mode=='ref':
        target_path = clean_ref_db
    else:
        target_path = clean_img_dir
    scores_list = []

    if mode =="perturbation":
        noise_img_name = os.path.basename(gen_i)
        clean_img_name = None
        for filename in os.listdir(target_path):
            if filename in noise_img_name:
                clean_img_name = filename
                break
        if clean_img_name == None:
            exit(f'clean img of {gen_i} not found,{os.listdir(target_path)} not in {noise_img_name}')
        clean_img_path = os.path.join(clean_img_dir,clean_img_name)
        img_gen = np.array(Image.open(gen_i).convert('RGB'))
        img_ref = np.array(Image.open(clean_img_path).convert('RGB'))
        
        # 统一图像尺寸（使用生成图像的尺寸）
        if img_gen.shape != img_ref.shape:
            img_ref = np.array(Image.fromarray(img_ref).resize((img_gen.shape[1], img_gen.shape[0])))
        scores = peak_signal_noise_ratio(img_ref, img_gen, data_range=255)
        return scores

    for filename in os.listdir(target_path):
        if not (filename.endswith('png') or filename.endswith('jpg')):
            continue
        clean_ref_img = os.path.join(target_path,filename)
        # 读取并预处理图像
        img_gen = np.array(Image.open(gen_i).convert('RGB'))
        img_ref = np.array(Image.open(clean_ref_img).convert('RGB'))
        
        # 统一图像尺寸（使用生成图像的尺寸）
        if img_gen.shape != img_ref.shape:
            img_ref = np.array(Image.fromarray(img_ref).resize((img_gen.shape[1], img_gen.shape[0])))
        scores = peak_signal_noise_ratio(img_ref, img_gen, data_range=255)
        scores_list.append(scores)
    # print(scores_list)
    # 计算PSNR（数值范围0-255）
    return np.mean(scores)

PSNR_Scorer = ScoreEval(func_get_score_of_one_image=PSNR_get_score)

def FID_get_score(gen_i, clean_ref_db=None, clean_img_dir=None, type_name="face", mode='ref'):
    assert mode in ['ref','ori']
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化 FID 模块
    fid = FrechetInceptionDistance(feature=2048).to(device)

    def load_images_from_dir(img_dir, real=True):
        for fname in os.listdir(img_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, fname)
                try:
                    img = read_image(img_path)  # dtype = torch.uint8, shape = [C, H, W]
                    img = transform(img).unsqueeze(0).to(device)
                    fid.update(img, real=real)
                except Exception as e:
                    print(f"Warning: Failed to load {img_path} - {e}")
    if mode == 'ori':
        load_images_from_dir(clean_img_dir, real=True)
    else:
        load_images_from_dir(clean_ref_db, real=True)
    load_images_from_dir(gen_i, real=False)
    return fid.compute().item()

def LPIPS_get_score(gen_i, clean_ref_db=None, clean_img_dir=None, type_name="face", mode='alex'):
    # attention! when culculating LPIPS, gen_i here we set as the perturbation image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=mode).to(device)
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # 根据需要调整图像大小
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1]范围
    ])

    names1 = sorted(os.listdir(gen_i))
    names2 = sorted(os.listdir(clean_img_dir))
    # print(names1)
    # print(names2)
    assert len(names1) == len(names2), "image number should be the same"

    scores = []
    for n1, n2 in zip(names1, names2):
        img1_path = os.path.join(gen_i, n1)
        img2_path = os.path.join(clean_img_dir, n2)

        img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
        img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            d = loss_fn(img1, img2)
        scores.append(d.item())

    mean_score = sum(scores) / len(scores)
    # print(f"平均 LPIPS 分数：{mean_score:.4f}")
    return mean_score

def SSIM_get_score(gen_i, clean_ref_db=None, clean_img_dir=None, type_name="face", mode=None):
    # attention! when culculating LPIPS, gen_i here we set as the perturbation image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    names1 = sorted(os.listdir(gen_i))
    names2 = sorted(os.listdir(clean_img_dir))
    # print(names1)
    # print(names2)
    assert len(names1) == len(names2), "image number should be the same"

    scores = []
    for n1, n2 in zip(names1, names2):
        img1_path = os.path.join(gen_i, n1)
        img2_path = os.path.join(clean_img_dir, n2)

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1 = transform(img1).permute(1, 2, 0).numpy()  # HWC
        img2 = transform(img2).permute(1, 2, 0).numpy()

        s = ssim(img1, img2, data_range=1.0, channel_axis=-1) 
        scores.append(s)

    mean_score = sum(scores) / len(scores)
    # print(f"平均 SSIM 分数：{mean_score:.4f}")
    return mean_score

def get_feature_list(clean_ref_dir, type_name):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    ref_img_names = [path for path in os.listdir(clean_ref_dir) if path.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # ref_img_path = os.path.join(clean_ref_dir, ref_img_names[0])
    # print(ref_img_path)
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_key="sk-003010ca5dd24e6a9f68af54420733a0",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    prompt_text = None
    with open(f'{script_dir}/prompts/{type_name}.txt', 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    final_features = []
    for ref_img_name in ref_img_names:
        ref_img_path = os.path.join(clean_ref_dir, ref_img_name)
        print(ref_img_path)
        base64_image = encode_image(ref_img_path)

        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # We can select model: https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[{"role": "user","content": [
                    {"type": "text","text": prompt_text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}, }
                    ]}]
            )
        raw_text = completion.choices[0].message.content
        if raw_text.startswith("{"):
            raw_text = raw_text[1:]
        if raw_text.endswith("}"):
            raw_text = raw_text[:-1]
        key_features = eval(raw_text)

        print(len(key_features))
        print(key_features)
        if ":" in key_features[0]:
            for i in range(len(key_features)):
                key_features[i] = key_features[i].split(":")[1]
        time.sleep(3)
        final_features+= key_features
    return final_features
    # return ['Impressionistic brushstrokes', 'Soft color palette used.', 'Dynamic ocean waves depicted.', 'Sailboat central focus point.']


def get_score(image_dir, clean_ref_dir=None, clean_img_dir = None, type_name="person", perturbed_img_dir=None, eval_items=None):
    # 此处增加扰动图像的评估
    if type_name == "person":
        type_name = "face"
        
    result_dict = {}
    if type_name == "face":
        if 'SDS' in eval_items: result_dict['SDS'] = FDSR_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
        if 'CLIP_Face_IQA' in eval_items: result_dict['CLIP_Face_IQA'] = CLIP_Face_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
        if 'LIQE_Scene_Human' in eval_items: result_dict['LIQE_Scene_Human'] =  LIQE_Scene_Human_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    else:
        if 'SDS' in eval_items: CLIP_zero_short_classification_score = CLIP_zero_short_classification_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
        if 'SDS' in eval_items: result_dict['SDS'] = CLIP_zero_short_classification_score
         
    if 'CLIPIQA' in eval_items: result_dict['CLIPIQA'] = CLIP_IQA_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    if 'BRISQUE' in eval_items: result_dict['BRISQUE'] = BRISQUE_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    if 'LIQE_Quality' in eval_items: result_dict['LIQE_Quality'] = LIQE_Quality_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    if 'IMS_CLIP_ViT-B/32' in eval_items: result_dict['IMS_CLIP_ViT-B/32'] = IMS_CLIP_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    if 'CLIP_IQAC' in eval_items: result_dict['CLIP_IQAC'] = CLIP_IQAC_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    if 'PSNR_ref' in eval_items: result_dict['PSNR_ref'] = PSNR_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, clean_image_dir=clean_img_dir, type_name=type_name, mode='ref')
    if 'PSNR_train' in eval_items: result_dict['PSNR_train'] = PSNR_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, clean_image_dir=clean_img_dir, type_name=type_name, mode='ori')
    if 'PSNR_perturbation' in eval_items: result_dict['PSNR_perturbation'] = PSNR_Scorer(gen_image_dir=perturbed_img_dir, clean_ref_db=clean_ref_dir, clean_image_dir=clean_img_dir, type_name=type_name, mode='perturbation')
    if 'FID_ref' in eval_items: result_dict['FID_ref'] = FID_get_score(gen_i=image_dir, clean_ref_db=clean_ref_dir, clean_img_dir=clean_img_dir, type_name=type_name, mode='ref')
    if 'FID_train' in eval_items: result_dict['FID_train'] = FID_get_score(gen_i=image_dir, clean_ref_db=clean_ref_dir, clean_img_dir=clean_img_dir, type_name=type_name, mode='ori')
    if 'LPIPS' in eval_items: result_dict['LPIPS'] = LPIPS_get_score(gen_i=perturbed_img_dir, clean_ref_db=None, clean_img_dir=clean_img_dir, type_name=type_name, mode='alex')
    if 'SSIM' in eval_items: result_dict['SSIM'] = SSIM_get_score(gen_i=perturbed_img_dir, clean_ref_db=None, clean_img_dir=clean_img_dir, type_name=type_name)
    if 'CLIP_ISIMC' in eval_items: result_dict['CLIP_ISIMC'] = CLIP_ISIMC_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name, mode=get_feature_list(clean_ref_dir,type_name))
    if type_name == "face":
        if 'IMS_VGG-Face_cosine' in eval_items: result_dict[f"IMS_VGG-Face_cosine"] = IMS_Face_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name,mode=["cosine","VGG-Face"])
                
    return result_dict


if __name__ == "__main__":
    pass