# From LLMs to LRMs: Rethinking Pruning for Reasoning-Centric Models

[![arXiv](https://img.shields.io/badge/arXiv-2601.18091-b31b1b.svg)](https://arxiv.org/abs/2601.18091)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://arxiv.org/abs/2601.18091)
[![Code](https://img.shields.io/badge/GitHub-EIT--NLP%2FLRM--Pruning-blue.svg)](https://github.com/EIT-NLP/LRM-Pruning)

A unified evaluation framework for **structure-pruned LLMs** across **classification, generation, and reasoning**.
![ARC Benchmark](arc.png)

## Why this repo

Most pruning work targets **instruction-following LLMs (LLM-instruct)**, but it is unclear whether these pruning recipes transfer to **reasoning-augmented models (LLM-think)** that explicitly generate long intermediate reasoning traces.

This repo provides a **controlled and unified** evaluation pipeline:
- Aligns **pruning calibration** and **post-pruning recovery** data with each model’s original training distribution to obtain more stable pruning behavior. 
- Evaluates **static depth**, **static width**, and **dynamic depth** pruning across **17 tasks** spanning classification, generation, and reasoning. 

---

## Main findings 

Empirical takeaways you can expect to reproduce here:
- **Depth vs. width is task-dependent**: depth pruning tends to be better on **classification**, while width pruning is often more robust on **generation & reasoning**.
- **Static vs. dynamic is paradigm-dependent**: dynamic pruning is strong on LLM-instruct classification/generation, but can be fragile for **long-chain reasoning** in LLM-think; static pruning often preserves reasoning better.

---

## Supported pruning methods

We group structured pruning methods into three categories.

### Static Depth

- **SLEB**  
  [![arXiv](https://img.shields.io/badge/arXiv-2402.09025-b31b1b.svg)](https://arxiv.org/abs/2402.09025)
  [![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/jiwonsong-dev/SLEB)
- **ShortGPT**  
  [![arXiv](https://img.shields.io/badge/arXiv-2403.03853-b31b1b.svg)](https://arxiv.org/abs/2403.03853)
  [![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/sramshetty/ShortGPT/tree/hf-models)
- **Shortened-PPL / Shortened-Taylor**  
  [![arXiv](https://img.shields.io/badge/arXiv-2402.02834-b31b1b.svg)](https://arxiv.org/abs/2402.02834)
  [![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/Nota-NetsPresso/shortened-llm)

### Static Width
- **LLM-Pruner**  
  [![arXiv](https://img.shields.io/badge/arXiv-2305.11627-b31b1b.svg)](https://arxiv.org/abs/2305.11627)
  [![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/horseee/LLM-Pruner)
- **SliceGPT**  
  [![arXiv](https://img.shields.io/badge/arXiv-2401.15024-b31b1b.svg)](https://arxiv.org/abs/2401.15024)
  [![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/microsoft/TransformerCompression)

### Dynamic
- **MOD**  
  [![arXiv](https://img.shields.io/badge/arXiv-2404.02258-b31b1b.svg)](https://arxiv.org/abs/2404.02258)
- **D-LLM**  
  [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202024-red.svg)](https://neurips.cc/virtual/2024/poster/94977)
- **SkipGPT**  
  [![arXiv](https://img.shields.io/badge/arXiv-2506.04179-b31b1b.svg)](https://arxiv.org/abs/2506.04179)
  [![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/EIT-NLP/SkipGPT)
  
We are continuously integrating more structured pruning methods to expand coverage.

---

## Benchmarks

We evaluate pruned models on **instruction following** and **reasoning**.

### Instruction-following (LLM-instruct)

Classification:
- BoolQ, PIQA, HellaSwag, WinoGrande, ARC-E/ARC-C, OpenBookQA

Generation:
- IFEval, TruthfulQA, PopQA, HumanEval, HumanEval+

### Reasoning (LLM-think)
- AIME 2024, MATH-500, LiveCodeBench, GPQA-Diamond, JEEBench

> Note: the paper evaluates 17 tasks in total across these categories. 
---

## Quickstart

### Installation
```bash
git clone https://github.com/EIT-NLP/LRM-Pruning.git
cd LRM-Pruning

# TODO: recommend a pinned environment
conda create -n lrm-pruning python=3.10 -y
conda activate lrm-pruning

pip install -r requirements.txt
```
# Evaluation
We modify the implementation of the `models` package in **lm_eval** so that pruning methods that change the model architecture can be directly evaluated. Specifically, we extend the interface in `huggingface.py` by introducing several additional arguments for loading pruned checkpoints or adapters.

With these modifications, the following pruning methods can be evaluated by specifying the corresponding parameters in `model_args`.

| Pruning Method | Parameters |
|---|---|
| LLM-Pruner (w/o LoRA) | `prune_model_path` |
| LLM-Pruner (w LoRA) | `prune_model_path`, `tune_model_path` |
| SliceGPT (w/o LoRA) | `slice_model_path`, `sparsity` |
| SliceGPT (w LoRA) | `slice_model_path`, `adapter_path`, `sparsity` |
| MOD | `mod_ckpt`, `sparsity` |
| D-LLM | `dllm_ckpt` |
| SkipGPT | `skipgpt_ckpt` |

Evaluation is performed using **lm-evaluation-harness**.

First activate the environment:

```bash
conda activate lrm-pruning
```

Define the dense model path:

```bash
DENSE_MODEL_PATH="models/Llama-3.1-Tulu-3-8B-SFT/"
```

Then specify the arguments for the pruning method.

Example configurations:

LLM-Pruner (without LoRA)

```bash
MODEL_ARGS="pretrained=${DENSE_MODEL_PATH},prune_model_path=${PRUNE_MODEL_PATH}"
```

LLM-Pruner (with LoRA)

```bash
MODEL_ARGS="pretrained=${DENSE_MODEL_PATH},prune_model_path=${PRUNE_MODEL_PATH},tune_model_path=${TUNE_MODEL_PATH}"
```

SliceGPT (without LoRA)

```bash
MODEL_ARGS="pretrained=${DENSE_MODEL_PATH},slice_model_path=${SLICE_MODEL_PATH},sparsity=${SPARSITY}"
```

SliceGPT (with LoRA)

```bash
MODEL_ARGS="pretrained=${DENSE_MODEL_PATH},slice_model_path=${SLICE_MODEL_PATH},adapter_path=${ADAPTER_PATH},sparsity=${SPARSITY}"
```

MOD

```bash
MODEL_ARGS="pretrained=${DENSE_MODEL_PATH},mod_ckpt=${MOD_CKPT},sparsity=${SPARSITY}"
```

D-LLM

```bash
MODEL_ARGS="pretrained=${DENSE_MODEL_PATH},dllm_ckpt=${DLLM_CKPT}"
```

SkipGPT

```bash
MODEL_ARGS="pretrained=${DENSE_MODEL_PATH},skipgpt_ckpt=${SKIPGPT_CKPT}"
```

After setting `MODEL_ARGS`, run the evaluation.

### Classification

```bash
lm_eval \
    --model hf \
    --model_args ${MODEL_ARGS} \
    --tasks <TASK_NAME> \
    --num_fewshot 0 \
    --batch_size auto
```

### Generation

```bash
olmes \
    --model ${MODEL_ARGS} \
    --task <TASK_NAME> \
    --batch-size auto \
    --output-dir <output-dir> \
    --cached-output-dir
```

### Reasoning

```bash
python -m eval.eval \
    --model hf \
    --model_args ${MODEL_ARGS} \
    --task <TASK_NAME> \
    --batch_size auto \
    --output_path <output_path>
```

In practice, users only need to modify `MODEL_ARGS` according to the pruning method being evaluated.
