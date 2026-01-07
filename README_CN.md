# HY-Video-PRFL

<div align="center">
  <img src="assets/logo.svg"  height=100>

# ⚡ HY-Video-PRFL: 视频生成模型是优秀的潜在奖励模型

</div>

视频生成模型既能创造也能评估——我们使14B模型能够在67GB显存内完成完整的720P×81帧后训练，相比传统方法实现1.5倍的速度提升和56%的运动质量改进。

<div align="center">
  <a href="https://github.com/Tencent-Hunyuan/HY-Video-PRFL"><img src="https://img.shields.io/static/v1?label=HY-Video-PRFL%20Code&message=Github&color=blue"></a> &ensp;
  <a href="https://hy-video-prfl.github.io/HY-VIDEO-PRFL/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/pdf/2511.21541"><img src="https://img.shields.io/badge/ArXiv-2511.21541-red"></a> &ensp;
</div>

<br>

![image](assets/teaser.jpg)

> [**HY-Video-PRFL: 视频生成模型是优秀的潜在奖励模型**](https://arxiv.org/pdf/2511.21541)

## 🔥🔥🔥 最新动态！

* **2025年12月7日**: 👋 我们发布了HY-Video-PRFL的训练和推理代码。
* **2025年11月26日**: 👋 我们发布了论文和项目主页。[[论文](https://arxiv.org/pdf/2511.21541)] [[项目主页](https://hy-video-prfl.github.io/HY-VIDEO-PRFL/)]

## 📑 开源计划

- HY-Video-PRFL
  - [x] PAVRM的训练和推理代码
  - [x] PRFL的训练和推理代码
  
## 📋 目录

- [🔥🔥🔥 最新动态！](#-最新动态)
- [📑 开源计划](#-开源计划)
- [📖 摘要](#-摘要)
- [🏗️ 模型架构](#️-模型架构)
- [📊 性能表现](#-性能表现)
- [🎬 案例展示](#-案例展示)
- [📜 环境要求](#-环境要求)
- [🛠️ 安装](#️-安装)
- [🧱 下载模型](#-下载模型)
- [🎓 训练](#-训练)
- [🚀 推理](#-推理)
- [📝 引用](#-引用)
- [🙏 致谢](#-致谢)

## 📖 摘要

奖励反馈学习（ReFL）已被证明能有效地使图像生成与人类偏好保持一致。然而，将其扩展到视频生成面临着重大挑战。现有的视频奖励模型依赖于为像素空间输入设计的视觉-语言模型，在计算成本高昂的VAE解码后，将ReFL优化限制在接近完成的去噪步骤。

**HY-Video-PRFL** 引入了**过程奖励反馈学习（PRFL）**，这是一个完全在潜在空间中进行偏好优化的框架。我们证明了预训练的视频生成模型天然适合在噪声潜在空间中进行奖励建模，使得能够在整个去噪链中进行高效的梯度反向传播，而无需VAE解码。

**核心优势:**
- ✅ 高效的潜在空间优化
- ✅ 显著的内存节省
- ✅ 相比RGB ReFL快1.4倍的训练速度
- ✅ 更好地与人类偏好保持一致

## 🏗️ 模型架构

![image](assets/method.png)

**传统的RGB ReFL** 依赖于为像素空间输入设计的视觉-语言模型，需要昂贵的VAE解码，并将优化限制在后期去噪步骤。

**我们的PRFL方法** 利用预训练的视频生成模型作为噪声潜在空间中的奖励模型。这实现了：
- 无需VAE解码的全链梯度反向传播
- 针对运动动态和结构一致性的早期监督
- 大幅减少内存消耗和训练时间

## 📊 性能表现

### 定量结果

我们的实验表明，PRFL在运动质量方面实现了显著改进（动态度提升+56.00，人体解剖结构提升+21.52，并且与人类偏好有更好的对齐），同时在效率方面也取得了显著提升（至少快1.4倍的训练速度和显著的内存节省）。

#### 文本生成视频结果
![image](assets/T2V_exp.png)

#### 图像生成视频结果
![image](assets/I2V_exp.png)

#### 效率对比
<img src="assets/efficiency.png" width="50%">

## 🎬 案例展示

### 文本生成视频

|480P 分辨率|720P 分辨率|
|---|---|
|<video src="https://github.com/user-attachments/assets/eed8d875-4b0d-43ec-b013-f0d10c2e107a" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```Two shirtless men with short dark hair are sparring in a dimly lit room. They are both wearing boxing gloves, one red and one black. One man is wearing white shorts while the other is wearing black shorts. There are several screens on the wall displaying images of buildings and people.```</details>|<video src="https://github.com/user-attachments/assets/4871a9f9-9b15-4065-8680-1c8059242707" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```A woman with fair skin, dark hair tied back, and wearing a light green t-shirt is visible against a gray background. She uses both hands to apply a white substance from below her eyes upward onto her face. Her mouth is slightly open as she spreads the cream.```</details>|
|<video src="https://github.com/user-attachments/assets/27296444-973b-4815-b103-5a2ee06404db" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```The woman has dark eyes and is holding a black smartphone to her ear with her right hand. She is typing on the keyboard of an open silver laptop computer with her left hand. Her fingers have blue nail polish. She is sitting in front of a window covered by sheer white curtains.```</details>|<video src="https://github.com/user-attachments/assets/430ff1ca-63ae-4a67-b6fb-b010c7ceec29" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```A light-skinned man with short hair wearing a yellow baseball cap, plaid shirt, and blue overalls stands in a field of sunflowers. He holds a cut sunflower head in his left hand and touches it with his right index finger. Several other sunflowers are visible in the background, some facing away from the camera.```</details>|

### 图像生成视频

|480P 分辨率|720P 分辨率|
|---|---|
|<img src="assets/videos/more/14_seed_876367.jpg" width="600">|<img src="assets/videos/more/109_seed_677347.jpg" width="600">|
|<video src="https://github.com/user-attachments/assets/956f5f64-1680-45a0-8666-9fda8e253017" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```A monochromatic video capturing a cat's gaze into the camera```</details>|<video src="https://github.com/user-attachments/assets/8dfeb1ae-8b9c-45aa-899a-3b48903629f9" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```A young boy is jumping in the mud```</details>|
|<img src="assets/videos/more/artlist_video_60ca5873a5c21ff4e4785b1997239d87_seed_561368.jpg" width="600">|<img src="assets/videos/more/real_1246_seed_277973.jpg" width="600">|
<video src="https://github.com/user-attachments/assets/d6d48d0c-cca5-4bdc-95c7-6a2858679111" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```A family of four eats fast food at a table.```</details>| <video src="https://github.com/user-attachments/assets/f39d3382-5430-412d-999e-c4c108835b6c" width="600" controls autoplay loop></video> <details><summary>📋 展示提示词</summary>```Normal speed, Medium shot, Eye level angle, Third person viewpoint, Static camera movement, Frame-within-frame composition, Shallow depth of field, Natural light, Cinematic style, Desaturated palette with slate blue, dusty rose, and dark wood tones color palette, Dramatic atmosphere. The scene is set on a patio or veranda, framed by a stone archway. In the back, there is a large, weathered wooden gate set into a stone wall. Six people are gathered on a stone patio in front of a large wooden gate. On the right, two men are seated at a dark wooden table. An older man in a grey traditional jacket holds a cane and gestures with his right hand while speaking. A younger man in a light grey suit sits beside him, listening. On the left side of the frame, a man in a dark suit stands with his back to the camera. Next to him, a woman in a pink patterned cheongsam and a woman in a grey skirt suit are standing close together, whispering. The women then turn and smile towards the men at the table. The man in the dark suit turns to face the group, revealing a newborn baby cradled in his arms, wrapped in a pink blanket. He takes a few steps forward, holding the baby. The women look at him and the infant. The older man at the table continues to talk, now gesturing towards the man with the baby. The man holding the baby looks down at the infant as he continues to walk slowly. The table is set with white cups, plates, fruit, and a dark wooden box.```</details>|


## 📜 环境要求

### 硬件要求

我们建议使用至少80GB显存的GPU以获得更好的生成质量。

### 软件要求

* **操作系统**: Linux
* **CUDA**: 12.4

## 🛠️ 安装

### 步骤1: 克隆仓库
```bash
git clone https://github.com/Tencent-Hunyuan/HY-Video-PRFL.git
cd HY-Video-PRFL
```

### 步骤2: 设置环境

我们推荐使用CUDA 12.4版本进行安装。Conda的安装说明可在[这里](https://www.anaconda.com/docs/main)找到。

```bash
# 创建conda环境
conda create -n HY-Video-PRFL python==3.10

# 激活环境
conda activate HY-Video-PRFL

# 安装PyTorch和依赖项（CUDA 12.4）
pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# 安装额外依赖项
pip3 install git+https://github.com/huggingface/transformers qwen-vl-utils[decord]
pip3 install git+https://github.com/huggingface/diffusers
pip3 install xfuser -i https://pypi.org/simple
pip3 install flash-attn==2.5.0 --no-build-isolation
pip3 install -e .
pip3 install nvidia-cublas-cu12==12.4.5.8

export PYTHONPATH=./
```

## 🧱 下载模型

在训练或推理前下载预训练模型：

| 模型 | 分辨率 | 下载链接 | 说明 |
|-------|-----------|----------------|-------|
| **Wan2.1-T2V-14B** | 480P & 720P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) <br> 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | 文本生成视频模型 |
| **Wan2.1-I2V-14B-720P** | 720P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) <br> 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | 图像生成视频（高分辨率） |
| **Wan2.1-I2V-14B-480P** | 480P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) <br> 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | 图像生成视频（标准） |

首先，确保已安装huggingface CLI或modelscope CLI。
```
pip install -U "huggingface_hub[cli]"
pip install modelscope
```
然后，下载预训练的DiT和VAE检查点。例如，可以使用以下命令将720P I2V任务的WAN2.1检查点下载到默认的```./weights```目录。
```
hf download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./weights
```

## 🎓 训练

### 1️⃣ 单GPU数据预处理
```bash
python3 scripts/preprocess/gen_wanx_latent.py --config configs/pre_480.yaml
```

我们在```temp_data/videos```中提供了几个视频作为模板训练数据，以及用于预处理的输入json文件```temp_data/temp_input_data.json```模板。```configs/pre_480.yaml```用于480P潜在提取，```configs/pre_720.yaml```用于720P。配置文件中的```json_path```和```save_dir```可以根据自己的训练数据自定义。

### 2️⃣ 数据标注和格式转换

奖励模型的标注（例如```"physics_quality": 1, "human_quality": 1```）应添加到数据元文件中（例如```temp_data/480/meta_v1/0004e625d5bcb80130e1ea3d204e2488_meta_v1.json```）。这样我们得到元文件列表```temp_data/temp_data_480.list```和```temp_data/temp_data_720.list```，可用于PAVRM和PRFL训练。

### 3️⃣ 多GPU并行PAVRM训练

例如，要使用8个GPU训练PAVRM，可以使用以下命令。

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29500 scripts/pavrm/train_pavrm.py --config configs/train_pavrm_i2v_720.yaml
```

配置文件中的```meta_file_list```和```val_meta_file_list```可以根据自己的训练和验证数据自定义。我们为不同设置（t2v或i2v，480P或720P）提供了几个配置文件。需要注意的是，我们使用ce损失训练PAVRM。要使用bt损失训练PAVRM，可以使用配置文件```configs/train_pavrm_bt_i2v_720.yaml```。

### 4️⃣ 多GPU并行PRFL训练

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29500 scripts/prfl/train_prfl.py --config configs/train_prfl_i2v_720.yaml
```

配置文件中的```meta_file_list```可以根据自己的训练数据自定义，配置文件中的```lrm_transformer_path```、```lrm_mlp_path```和```lrm_query_attention_path```用于从上一步获得的奖励模型。我们为不同设置（t2v或i2v，480P或720P）提供了几个配置文件。

## 🚀 推理

### 1️⃣ 多GPU并行PAVRM推理

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29500 scripts/pavrm/inference_pavrm.py --config configs/infer_pavrm_i2v_720.yaml
```

配置文件中的```val_meta_file_list```可以根据自己的推理数据自定义，配置文件中的```resume_transformer_path```、```resume_mlp_path```和```resume_query_attention_path```用于待测试的奖励模型。

### 2️⃣ 多GPU并行PRFL推理

PRFL推理与其基础模型（例如Wan2.1）完全相同。

```bash
export negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29500 scripts/prfl/inference_prfl.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 1 \
    --task "i2v-14B"\
    --ckpt_dir "weights/Wan2.1-I2V-14B-720P" \
    --lora_path "" \
    --lora_alpha 0 \
    --dataset_path "temp_data/temp_prfl_infer_data.json" \
    --negative_prompt "$negative_prompt" \
    --size "1280*720" \
    --frame_num 81 \
    --sample_steps 40 \
    --sample_guide_scale 5.0 \
    --sample_shift 5.0 \
    --teacache_thresh 0 \
    --save_folder outputs/infer/prfl_i2v_720 \
    --transformer_path <YOUR_CKPT_PATH> \
    --offload_model False
```

**参数说明:**
- `--dit_fsdp` `--t5_fsdp`: 启用FSDP以提高内存效率
- `--task`: "t2v-14B"或"i2v-14B"
- `--ckpt_dir`: 预训练检查点文件路径
- `--lora_path` `--lora_alpha`: LoRA检查点文件路径和加载权重比例
- `--dataset_path`: 推理数据集文件路径
- `--size`: 输出分辨率（"1280\*720"或"832\*480"）
- `--frame_num`: 生成的帧数（默认：81）
- `--sample_steps`: 推理步数（默认：40）
- `--sample_guide_scale`: 无分类器引导比例（默认：5.0）
- `--sample_shift`: 流偏移（默认：5.0）
- `--save_folder`: 保存生成视频的路径
- `--teacache_thresh`: 启用teacache
- `--transformer_path`: PRFL检查点文件路径
- `--offload_model`: 卸载到CPU以节省GPU内存

## 📝 引用

如果您发现**HY-Video-PRFL**对您的研究有用，请引用：
```bibtex
@article{mi2025video,
  title={Video Generation Models are Good Latent Reward Models},
  author={Mi, Xiaoyue and Yu, Wenqing and Lian, Jiesong and Jie, Shibo and Zhong, Ruizhe and Liu, Zijun and Zhang, Guozhen and Zhou, Zixiang and Xu, Zhiyong and Zhou, Yuan and Lu, Qinglin and Tang, Fan},
  journal={arXiv preprint arXiv:2511.21541},
  year={2025}
}
```

## 🙏 致谢

我们真诚感谢以下项目的贡献者：
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [ImageReward](https://github.com/THUDM/ImageReward)
- [Diffusers](https://github.com/huggingface/diffusers)
- [HuggingFace](https://huggingface.co)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

---

<div align="center">
  
**如果您觉得有帮助，请给这个仓库加星 ⭐！**

</div>
