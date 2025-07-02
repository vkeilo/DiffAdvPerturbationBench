import json
import os
import glob
# get script path
now_dir_path = os.path.dirname(os.path.abspath(__file__))
DAPB_path = os.path.dirname(now_dir_path)
print(f"DAPB_path: {DAPB_path}")
print(f"now_dir_path: {now_dir_path}")

def resolve_unique_wildcard_path(wild_path: str) -> str:
    parts = wild_path.strip(os.sep).split(os.sep)
    current_path = os.sep if wild_path.startswith(os.sep) else "."
    
    for part in parts:
        if '*' in part or '?' in part or '[' in part:
            # 构造当前这一层的通配路径
            candidate_pattern = os.path.join(current_path, part)
            matches = glob.glob(candidate_pattern)
            matches = [m for m in matches if os.path.isdir(m)]
            if len(matches) == 0:
                raise FileNotFoundError(f"No match found for pattern: {candidate_pattern}")
            elif len(matches) > 1:
                raise ValueError(f"Multiple matches found for pattern: {candidate_pattern}\nMatches: {matches}")
            current_path = matches[0]
        else:
            # 普通路径，直接拼接
            current_path = os.path.join(current_path, part)
            if not os.path.isdir(current_path):
                raise FileNotFoundError(f"Directory does not exist: {current_path}")
    
    return os.path.abspath(current_path)
special_text = [
    "vux", "jiq", "qem", "zod", "dul", "wex", "hob", "taf", "yib", "nuz",
    "gax", "fep", "cim", "pyk", "teb", "lom", "sir", "dap", "kex", "yoc",
    "mib", "zow", "ruk", "hif", "cun", "baj", "tox", "geq", "vyx", "qum",
    "rek", "sov", "hax", "zud", "dif", "koy", "wen", "jib", "rax", "miv",
    "pob", "lut", "seg", "yox", "cud", "vig", "bez", "nam", "foj", "xur"
]

# special_text = [
#     "vux", "jiq", "qem", "zod", "dul"
# ]
# taskname = "Mist_SD21_VGGFace2_random50_mytarget2e-1_dynamic_r12"
# exp_name = "Mist_SD21_VGGFace2_random50_mytarget2e-1_dynamic_r12"
# taskname = "SimAC_SD21_VGGFace2_r8_idx10_1id1atk"
taskname = "MultiATK_SD21_VGGFace2_r8_idx10_1idmatk"
# exp_name = "SimAC_SD21_VGGFace2_r8_idx10_1id1atk"

# /data/home/yekai/github/DiffAdvPerturbationBench/Algorithms/mist/exp_datas_output/Mist_SD21_VGGFace2_random50_mytarget11r_dynamic_r12
r = 8
Alg = "MultiATK"
round = "50"
target_exp_path = "/data/home/yekai/github/DiffAdvPerturbationBench/datasets/auxdatas/gen_aux_exps/MultiATK_SD21_VGGFace2_r8_idx10_1idmatk"
# class-person/class-artwork
class_data_path = os.path.join(DAPB_path,"class-person")
save_path = f"{now_dir_path}/concepts_lists"
if not os.path.exists(save_path):
    os.makedirs(save_path)

concepts_num = 10
# default atk all id
attacked_id = [i for i in range(concepts_num)]
# attacked_id = [1]
data_path = os.path.join(DAPB_path,"datasets/VGGFace2-10x64")

atk_data_path = f"{target_exp_path}/*_id?replaceidhere?_*r{str(r)}*/noise-ckpt/{round}"
final_list = []

for i in range(concepts_num):
    instance_prompt = f"a photo of {special_text[i]} person"
    class_prompt = "a photo of person"
    instance_data_dir = resolve_unique_wildcard_path(atk_data_path.replace('?replaceidhere?',str(i))) if i in attacked_id else os.path.join(data_path,str(i),"set_B")
    # print(instance_data_dir)
    
    class_data_dir = class_data_path
    tmp_concept = {
        "instance_prompt":      instance_prompt,
        "class_prompt":         class_prompt,
        "instance_data_dir":    instance_data_dir,
        "class_data_dir":       class_data_dir
    }
    final_list.append(tmp_concept)

with open(os.path.join(save_path,f"{taskname}.json"), "w") as f:
    json.dump(final_list, f, indent=4)