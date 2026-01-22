import os
import torch
from os.path import basename
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from lm_eval.models.router_attn_mlp import apply_router_attn_mlp
from lm_eval.models.router_all import apply_router_all
from lm_eval.models.mod_twice import apply_mod_twice
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
import argparse

def lora(model):
        lora_config = LoraConfig(
            r=16,  # 低秩矩阵的秩
            lora_alpha=32,
            target_modules=["q_proj", "v_proj","gate_proj"],  # 指定应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        

        return model


def eval_lora_base_model_import(device="cuda"):
    # checkpoint_path = "/code/SKIPGPT/results/Llama-3.1-Tulu-3-8B_tulu-3-sft-mixture_4096_2/original_lr_0.0004_cosine_warmup_0.1_LoRA_True/checkpoint-7000/"
    checkpoint_path = "/code/SKIPGPT/results/Llama-3.1-Tulu-3-8B_tulu-3-sft-mixture_4096_2/original_lr_0.0004_cosine_warmup_0.1_LoRA_True_no_grid_skip/test-openbookqa/checkpoint-3000/"
    # checkpoint_path = "/code/SKIPGPT/results/Llama-3.1-Tulu-3-8B_tulu-3-sft-mixture_4096_2/original_lr_0.0004_cosine_warmup_0.1_LoRA_True_no_grid_skip/test-openbookqa/checkpoint-min-train-loss/"
    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                "/code/SKIPGPT/models/Llama-3.1-Tulu-3-8B/",
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            )
    model = lora(model)  # model 无 merge
    print("测试 Lora 后的 base model!")
    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print(model_pth_path)
    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model.to(device)

def eval_lor_model_import(checkpoint_path, pretrained, device="cuda"):
    print("run eval_lor_model_import!")
    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            ).to(device)
    # apply router layer
    model = apply_router_attn_mlp(model)
    model = lora(model)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print(model_pth_path)
    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model.to(device)

def eval_lora_router_all_model_import(checkpoint_path, pretrained, device="cuda"):
    print("run eval_lor_model_import!")
    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            ).to(device)
    # apply router_all layer
    print("D-LLM: apply router_all layer!")
    model = apply_router_all(model)
    lora_config = LoraConfig(
            r=16,  # 低秩矩阵的秩
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # 指定应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
        )
    model = get_peft_model(model, lora_config)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print(model_pth_path)
    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model.to(device)

def eval_lora_mod_twice_model_import(checkpoint_path, pretrained, sparsity,device="cuda"):
    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            ).to(device)
    # apply router_all layer
    print("Mod: apply mod_twice layer!")
    model = apply_mod_twice(model,sparsity)
    lora_config = LoraConfig(
            r=8,  # 低秩矩阵的秩 
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],  # 指定应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
        )
    model = get_peft_model(model, lora_config)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print(model_pth_path)
    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model.to(device)


def remove_duplicate_prefix(s: str, prefix: str) -> str:
    parts = s.split(prefix)
    if len(parts) <= 1:
        return s
    return prefix + ''.join(parts[1:])

def rename_state_dict(checkpoint_path, device="cuda"):
    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                "/code/SKIPGPT/models/Llama-3.1-Tulu-3-8B/",
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            )

    # apply router layer
    model = apply_router_attn_mlp(model).to('cuda')
    model = lora(model)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth')
    print(model_pth_path)
    state_dict = torch.load(model_pth_path)

    new_state_dict = {}
    prefix = 'base_model.model.'
    for k, v in state_dict.items():
        new_key = remove_duplicate_prefix(k, prefix)
        new_state_dict[new_key] = v
    # 保存新的 state_dict
    torch.save(new_state_dict, model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    # model.eval()
    return 
