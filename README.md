<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24,30&height=230&section=header&text=Qwen3-8B%20Coding%20Fine-Tune%20⚡&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=QLoRA%20%7C%20OpenCodeInstruct%20%7C%20Unsloth%20%7C%20Kaggle%20Dual%20T4%20DDP&descAlignY=57&descSize=18" width="100%"/>

<!-- Animated Typing Text -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=23&duration=2800&pause=1000&color=A855F7&center=true&vCenter=true&width=750&lines=Fine-tuning+Qwen3-8B+on+2x+Tesla+T4+GPUs+%F0%9F%94%A5;QLoRA+%2B+SFT+%2B+PyTorch+DDP+via+torchrun;Benchmarked+on+HumanEval%2B+%26+MBPP%2B+%F0%9F%93%8A;Artifacts+Published+to+Hugging+Face+%F0%9F%A4%97" alt="Typing SVG" />

<br/>

<!-- Primary Badges -->
<a href="https://github.com/mudassar2224/Fine_tune_qwen3-8b-opencodeinstruct-finetune">
  <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>
<a href="https://www.kaggle.com/code/malikgt5/qwen3-8b-coding-fine-tune-opencodeinstruct-qlo">
  <img src="https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</a>
<a href="https://huggingface.co/Maliktg7/qwen3-8b-opencodeinstruct-merged">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Merged%20Model-FFD21E?style=for-the-badge"/>
</a>
<a href="https://huggingface.co/Maliktg7/qwen3-8b-opencodeinstruct-checkpoints">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-LoRA%20Adapter-FFD21E?style=for-the-badge"/>
</a>

<br/><br/>

<!-- Tech Stack Badges -->
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20DDP-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-2x%20Faster-8A2BE2?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-TRL%20%7C%20PEFT-FFD21E?style=flat-square)
![Kaggle](https://img.shields.io/badge/Hardware-2x%20Tesla%20T4-20BEFF?style=flat-square&logo=nvidia&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Code%20Studio-FF7C00?style=flat-square&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)

</div>

<br/>

---

## 🧭 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [⚡ Key Features](#-key-features)
- [🏗️ Pipeline Architecture](#️-pipeline-architecture)
- [🧠 Engineering Decisions & Trade-offs](#-engineering-decisions--trade-offs)
- [📊 Benchmark Results](#-benchmark-results)
- [🐛 Critical Bug Fix: Unsloth Trainer Patch](#-critical-bug-fix-unsloth-trainer-patch)
- [💻 Fast Inference Quickstart](#-fast-inference-quickstart)
- [🎮 Gradio Code Studio UI](#-gradio-code-studio-ui)
- [🛠️ Tech Stack & Hardware](#️-tech-stack--hardware)
- [📁 Repository Structure](#-repository-structure)
- [🚀 How to Reproduce](#-how-to-reproduce)
- [🗺️ Roadmap](#️-roadmap)
- [📜 License & Acknowledgments](#-license--acknowledgments)

<br/>

---

## 🎯 Project Overview

<img align="right" width="170" src="https://raw.githubusercontent.com/Tencent/VectorEngine/main/docs/assets/logo.png" />

This repository contains the complete end-to-end fine-tuning pipeline, evaluation harness, and interactive code generation studio for **Qwen3-8B**. 

The base model was fine-tuned into a coding specialist using **QLoRA + SFT** on a quality-filtered slice of [NVIDIA's OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) dataset on Kaggle's **Dual Tesla T4 GPUs** using PyTorch **Distributed Data Parallelism (DDP)** via `torchrun`.

* **GitHub Repository:** [Fine_tune_qwen3-8b-opencodeinstruct-finetune](https://github.com/mudassar2224/Fine_tune_qwen3-8b-opencodeinstruct-finetune)
* **Kaggle Notebook:** [qwen3-8b-coding-fine-tune-opencodeinstruct-qlo](https://www.kaggle.com/code/malikgt5/qwen3-8b-coding-fine-tune-opencodeinstruct-qlo)
* **Hugging Face Merged Model (fp16):** [Maliktg7/qwen3-8b-opencodeinstruct-merged](https://huggingface.co/Maliktg7/qwen3-8b-opencodeinstruct-merged)
* **Hugging Face LoRA Adapter:** [Maliktg7/qwen3-8b-opencodeinstruct-checkpoints](https://huggingface.co/Maliktg7/qwen3-8b-opencodeinstruct-checkpoints)

<br clear="right"/>

<br/>

---

## ⚡ Key Features

* 🚀 **VRAM Efficient QLoRA:** Memory footprint reduced from ~16 GB VRAM down to **~5.5 GB VRAM** using **Unsloth 4-bit quantization**.
* 🌊 **Streamed & Deduplicated Pipeline:** Streams NVIDIA's 5M-example OpenCodeInstruct dataset without full download, applying quality thresholds (`avg_test_score >= 0.8`) and exact-hash deduplication.
* ⚡ **Multi-GPU DDP Acceleration:** Configured with `torchrun --nproc_per_node=2` to utilize Dual Tesla T4 GPUs smoothly.
* 🧪 **EvalPlus Benchmarking:** Evaluated zero-shot `pass@1` performance against base models on **HumanEval+** and **MBPP+**.
* 🎨 **Interactive Code Studio UI:** Features a custom Gradio web dashboard with developer persona presets, generation parameters, and single-GPU routing (`cuda:0`).

<br/>

---

## 🏗️ Pipeline Architecture

```mermaid
flowchart LR
    A["🧊 Qwen3-8B-Instruct<br/>4-bit (Unsloth)"] --> B["📚 OpenCodeInstruct<br/>(Streamed Dataset)"]
    B --> C["🧹 Quality Filter<br/>avg_test_score ≥ 0.8"]
    C --> D["🔁 Hash Deduplication"]
    D --> E["✂️ Sequence Length Filter<br/>≤2048 tokens"]
    E --> F["🎯 QLoRA (r=16, α=32)<br/>SFT via TRL"]
    F --> G["⚡ DDP via torchrun<br/>2× Tesla T4 GPUs"]
    G --> H["💾 Checkpoint every 200 steps<br/>→ Pushed to 🤗 Hub"]
    H --> I{"More Training<br/>Sessions?"}
    I -- Yes --> G
    I -- No --> J["🧬 Merge Weights to fp16<br/>save_pretrained_merged"]
    J --> K["🚀 Published to<br/>🤗 Hugging Face"]
    K --> L["🎨 Gradio Code Studio UI"]

    style A fill:#6366f1,color:#fff
    style F fill:#a855f7,color:#fff
    style G fill:#ec4899,color:#fff
    style K fill:#f59e0b,color:#fff
    style L fill:#10b981,color:#fff
