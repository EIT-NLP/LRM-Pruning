import  torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple, Any
from transformers import PreTrainedModel, DynamicCache, Cache

class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        original_type = x.dtype
        self.weight_predictor.to(torch.float32)
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype)).squeeze(
            -1
        )  # [batch_size, seq_len]
        return weights.to(original_type)
    
class mod_twice_llama (nn.Module):
    def __init__(self, hidden_size,block,sparsity = 0.2):
        super().__init__()
        self.router = TokenRouter(hidden_size)
        self.router_mlp = TokenRouter(hidden_size)
        self.block = block
        self.capacity = 1 - sparsity  # 0.2 means 80% of tokens are selected
        self.training_step = 0

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_value: Optional[DynamicCache],
                output_attentions: bool,
                use_cache: bool,
                cache_position: Optional[torch.Tensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs: Any
                ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        b, s, d = hidden_states.shape

        # 计算attention mask
        weights = self.router(hidden_states)

        # if self.router.training:
        #     if self.training_step < self.args.gradient_accumulation_steps * self.args.max_steps_stage2:
        #         self.training_step += 1 
        #     self.capacity = self.args.capacity + ((1 - self.args.capacity) * (1. / ((self.training_step-1)//self.args.gradient_accumulation_steps+1)))

        if torch.isnan(hidden_states).any():
            warnings.warn(
                "NaN detected in input tokens, this is not intended to happen, please check your model. Before retraining, you could try the model with flash-attn-2 enabled.")

        k = max(1, int(self.capacity * s)) # 压缩 capacity
        top_k_values, _ = torch.topk(weights, k, dim=1, sorted=True)
        threshold = top_k_values[:, -1]
        selected_mask = weights > threshold.unsqueeze(-1) if k > 1 else weights >= threshold.unsqueeze(-1)
        # import ipdb; ipdb.set_trace()
        # # 合并attention_mask和selected_mask
        # attention_mask_bool=attention_mask.bool()
        # combined_mask = attention_mask_bool & selected_mask

        
        # 经过block的atttention layer
        residual = hidden_states
        hidden_states = self.block.input_layernorm(hidden_states)

        hidden_states, self_attn_weights,present_key_value = self.block.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states= hidden_states * weights.unsqueeze(-1)

        hidden_states = residual + hidden_states
        hidden_states = torch.where(selected_mask.unsqueeze(-1), hidden_states, residual)
        
        # 计算mlp mask
        weights_mlp = self.router_mlp(residual)

        top_k_values_mlp, _ = torch.topk(weights_mlp, k, dim=1, sorted=True)
        threshold_mlp = top_k_values_mlp[:, -1]
        selected_mask_mlp= weights_mlp > threshold_mlp.unsqueeze(-1) if k > 1 else weights_mlp >= threshold_mlp.unsqueeze(-1)
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.block.post_attention_layernorm(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        hidden_states = hidden_states * weights_mlp.unsqueeze(-1)
        hidden_states = residual + hidden_states
        hidden_states = torch.where(selected_mask_mlp.unsqueeze(-1), hidden_states, residual)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
def apply_mod_twice(model: PreTrainedModel, sparsity) -> PreTrainedModel:
    hidden_size = model.config.hidden_size
    new_layers = nn.ModuleList()
    if model.__class__.__name__ == "LlamaForCausalLM":
        for i, layer in enumerate(model.model.layers):
            if i==0 or i==len(model.model.layers)-1:
                new_layer = layer
            else:
                new_layer = mod_twice_llama(hidden_size,layer,sparsity)
            new_layers.append(new_layer)

    model.model.layers = new_layers 
    class_name = model.__class__.__name__

    # Insert MoD before the For
    if 'For' in class_name:
        parts = class_name.split('For', 1)
        modified_class_name = parts[0] + 'MoDFor' + parts[1]
    else:
        modified_class_name = 'MoD' + class_name  # If it doesn't find any i prepends MoD

    model.__class__.__name__ = modified_class_name

    return model