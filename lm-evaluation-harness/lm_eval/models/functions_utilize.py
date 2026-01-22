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

# SkipGPT_Evauation
def SkipGPT_Evauation(checkpoint_path, pretrained, device="cuda"):
    print("Load Checkponit for SkipGPT_Evauation!")

    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            ).to(device)

    # apply router layer
    model = apply_router_attn_mlp(model)

    # lora
    model = lora(model)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print("Checkpoint Path:", model_pth_path)

    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model

# D-LLM_Evaluation
def DLLM_Evaluation(checkpoint_path, pretrained, device="cuda"):
    print("Load Checkponit for D-LLM_Evaluation!")
    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            ).to(device)

    # apply router_all layer
    print("D-LLM: apply router_all layer!")
    model = apply_router_all(model)

    # LoRA
    lora_config = LoraConfig(
            r=16,  # 低秩矩阵的秩
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # 指定应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
        )
    model = get_peft_model(model, lora_config)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print("Checkpoint Path:", model_pth_path)

    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model

# MOD_Evaluation
def MOD_Evaluation(checkpoint_path, pretrained, sparsity,device="cuda"):
    print("Load Checkponit for MOD_Evaluation!")

    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            ).to(device)

    # apply router_all layer
    print("Mod: apply mod_twice layer!")
    model = apply_mod_twice(model,sparsity)

    # LoRA
    lora_config = LoraConfig(
            r=16,  # 低秩矩阵的秩
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # 指定应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
        )
    model = get_peft_model(model, lora_config)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print("Checkpoint Path:", model_pth_path)

    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model

# Dense_LoRA Evaluation
def Dense_LoRA(checkpoint_path, pretrained,device="cuda"):
    print("Load Checkponit for Dense_LoRA Evaluation!")

    attn_mode = "flash_attention_2" if device == "cuda" else None  # 自动切换
    model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_mode,
            ).to(device)

    # LoRA
    model = lora(model)  # model 无 merge

    model_pth_path = os.path.join(checkpoint_path, 'model.pth') # post bin else pth
    print("Checkpoint Path:", model_pth_path)

    state_dict = torch.load(model_pth_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Missing keys: {missing_keys}")
    print(f"[Info] Unexpected keys: {unexpected_keys}")
    model.eval()
    return model
