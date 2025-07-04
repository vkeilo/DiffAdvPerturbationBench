# DiffAdvPerturbationBench

**DiffAdvPerturbationBench** is an evaluation framework for unifying the assessment of various perturbation-based defense algorithms in the context of customized diffusion models (e.g., DreamBooth fine-tuning).

This project focuses on evaluating a series of published image perturbation defense methods (e.g., Anti-Dreambooth, Mist, AdvDM, MetaCloak, PhotoGuard, SDS, SimAC) for their protection performance under **Stable Diffusion** fine-tuning tasks. It supports automated execution of multiple algorithms, perturbation generation, model fine-tuning, and image quality analysis.

---

## 🌐 Project Module Structure

The project mainly contains the following modules (`finished_test/` is a temporary directory):

```
DiffAdvPerturbationBench/
├── Algorithms/         # Various perturbation defense algorithms, all forked from official implementations and wrapped with a unified interface
├── train_and_gen/      # Modules for customized model training and image generation such as DreamBooth
├── eval/               # Modules for image evaluation metrics (CLIP-IQA/LPIPS/PSNR/SSIM/CLIP-IQAC, etc.)
├── finished_test/      # Temporary directory for caching experiment results (can be ignored)
```

---

## 📌 Supported Perturbation Defense Algorithms

The following mainstream image protection methods are currently supported within the framework:

- [x] Anti-Dreambooth
- [x] Mist
- [x] AdvDM
- [x] MetaCloak
- [x] PhotoGuard
- [x] SDS
- [x] SimAC

Each algorithm is placed in the `Algorithms/<AlgorithmName>/` directory and supports batch task execution and output saving.

---

## 🚀 Example Workflow (Using SimAC as an Example)

The following demonstrates the complete process from dataset and parameter configuration to perturbation generation:

#### VGGFace2 Dataset Preparation

VGGFace2:
CelebA:
WikiArt:

unzip to path DiffAdvPerturbationBench/datasets

### 1. Modify SimAC Configuration

Enter the corresponding algorithm directory and modify the experiment parameters:

```bash
cd DiffAdvPerturbationBench/Algorithms/SimAC
vim run_in_batch/gen_test_json.py
```

Example experiment parameters:

```python
test_lable = "SimAC_VGGFace2_random50_r4p8p12p16_test"

params_options = {
    "data_path": ["<ABS_PATH>/datasets"],
    "dataset_name": ["VGGFace2-clean"],
    "data_id": [i for i in range(50)],
    "r": [4, 8, 12, 16],
    "attack_steps": [100],
    "mixed_precision": ["bf16"],
    "model_path": ["/path/to/stable-diffusion-2-1-base"],
    "class_data_dir": ["/path/to/datasets/class-person"],
    "instance_prompt": ["a photo of sks person"],
    "class_prompt": ["a photo of a person"],
    "report_to": ["wandb"],
    "WANDB_MODE": ["offline"],
}
```

### 2. Construct JSON Configuration File for Generation Task

```bash
python run_in_batch/gen_test_json.py
```

This will generate the perturbation task list file `SimAC_VGGFace2_random50_r4p8p12p16_test.json` in `Algorithms/SimAC/run_in_batch`.

### 3. Perturbed Image Generation

On multi-GPU devices, use `device_n` to specify which GPU to use:

```bash
cd ..
python run_in_batch/run_in_batch.py --target SimAC_VGGFace2_random50_r4p8p12p16_test.json --device_n 0
```

### 4. View Output Results

After task completion, perturbation results will be saved in:

```
Algorithms/SimAC/exp_outputs/SimAC_VGGFace2_random50_r4p8p12p16_test/
```

Each subdirectory will contain:

- Clean original image
- Class reference image
- Perturbed image

### 5. Customized Generation on Perturbed Samples

- To generate a customized task list, edit the following content in `DiffAdvPerturbationBench/train_and_gen/create_json_dreambooth.py`:

```python
test_lable = "SimAC_SD21_VGGFace2_random50_r4p8p12p16"

params_options = {
    "DPAB_path":["path/to/DiffAdvPerturbationBench"],
    # VGGFace2-clean/wikiart-data
    "dataset_name":["VGGFace2-clean"],
    "exp_batch_path": ["Algorithms/SimAC/exp_datas_output/SimAC_VGGFace2_random50_r4p8p12p16_test"],
    "pretrained_model_name_or_path": ["path/to/stable-diffusion-2-1-base"],
    "wandb_project_name": ["Dreambooth_train"],
    "mixed_precision": ["fp16"],
    "dreambooth_training_steps": [2000],
    "db_lr":[5e-7],
    # "a photo of sks person"/"a painting of sks artwork"
    "instance_prompt": ["a photo of sks person"],
    "inference_prompts": ["a photo of sks person"],
    # painting/person
    "class_name": ["person"],
    # num of gen img use instance_prompt
    "eval_gen_img_num": [16],
    # aspl:50 fsmg:100 SimAC:50 mist:100 sds:100 metacloak:final
    "use_sample_steps": ["50"],
}
```

Generate task list:
```bash
python create_json_dreambooth.py
```
This will generate the task list file `SimAC_SD21_VGGFace2_random50_r4p8p12p16.json` under `DiffAdvPerturbationBench/train_and_gen`.

- DreamBooth-based customized training and image generation based on perturbed samples (using environment MetaCloakp):

```bash
python train_and_gen/db_train_in_batch.py --target SimAC_SD21_VGGFace2_random50_r4p8p12p16.json --device_n 0
```

This will add a customized image generation directory `train_output` in each perturbation result folder under `Algorithms/SimAC/exp_outputs/SimAC_VGGFace2_random50_r4p8p12p16_test`.

---

## 📊 Image Evaluation

Generated images will be uniformly processed by the `eval/` module. The following quality and similarity metrics are supported:

- CLIP-IQA
- CLIP-IQAC
- LPIPS
- PSNR / SSIM
- FID (optional)

Evaluation code (based on environment MetaCloakp):

```bash
python eval eval_genpics_batch.py --target your_abs_path_to/Algorithms/SimAC/exp_outputs/SimAC_VGGFace2_random50_r4p8p12p16_test --round 50 --device_n 0 --type face --prompt a_photo_of_sks_person
```

Here, `type` is the class name (supports `face`, `artwork`), and `round` and `prompt` correspond to the folder structure in `Algorithms/SimAC/exp_outputs/SimAC_VGGFace2_random50_r4p8p12p16_test`.  
`round` represents the perturbation attack steps, and `prompt` is the prompt used during the attack.  
The evaluation result `SimAC_VGGFace2_random50_r4p8p12p16_test.csv` is saved in `DiffAdvPerturbationBench/eval/eval_result`.

- The full process for other algorithms only differs in the parameter settings during perturbation task list generation. For details, refer to the original project fork of each algorithm.

---

## 📦 Environment Dependencies

CUDA configuration used on the experiment server:  
cuda: 11.8  
cudnn: 9.0.3

All experiment code requires only two conda environments.  
Except for the perturbation generation of Mist and Diff-Protect (AdvDM, SDS, PhotoGuard) algorithms which require the `mist` environment, all other operations are carried out in the `MetaCloakp` environment.

#### MetaCloakp  
https://github.com/VinAIResearch/Anti-DreamBooth

#### mist  
https://github.com/psyker-team/mist

---

## 📝 Notes

- All protection algorithms are from their original papers and have been adapted and uniformly wrapped.
- The fine-tuning process is based on DreamBooth and can be flexibly customized depending on the dataset (e.g., VGGFace2).

---

## 📬 Contact

If you have questions, feel free to open an issue or contact yecup1056@gmail.com.