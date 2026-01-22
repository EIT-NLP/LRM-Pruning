# inspired by  https://github.com/kyegomez/Mixture-of-Depths
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

from transformers import PreTrainedModel, DynamicCache, Cache
class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 直接从输入维度到输出权重预测
        self.weight_predictor = nn.Linear(embed_dim, 2)
        
        # 使用 He Kaiming 初始化
        nn.init.kaiming_uniform_(self.weight_predictor.weight, nonlinearity='linear')
        
        # 初始化 bias 为 0
        if self.weight_predictor.bias is not None:
            nn.init.zeros_(self.weight_predictor.bias)

    def forward(self, x):
        # 保存输入的原始数据类型
        original_type = x.dtype
        
        # 计算权重预测
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype))
        
        return weights.to(original_type)
    
# class TokenRouter(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         # 中间层的维度是 embed_dim 的四分之一
#         intermediate_dim = embed_dim //4
#         # 增加一个中间层
#         self.hidden_layer = nn.Linear(embed_dim, intermediate_dim)
#         self.relu = nn.ReLU() 
#         self.weight_predictor = nn.Linear(intermediate_dim, 2)
        
#         # 使用 He Kaiming 初始化
#         nn.init.kaiming_uniform_(self.hidden_layer.weight, nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.weight_predictor.weight, nonlinearity='linear')

#         # 初始化 bias 为 0
#         if self.hidden_layer.bias is not None:
#             nn.init.zeros_(self.hidden_layer.bias)
#         if self.weight_predictor.bias is not None:
#             nn.init.zeros_(self.weight_predictor.bias)

#     def forward(self, x):
#         original_type = x.dtype
        
#         # 先通过中间层并激活，再传递到 weight_predictor
#         x = self.hidden_layer(x.to(self.hidden_layer.weight.dtype))
#         x = self.relu(x)  # 使用 ReLU 激活函数
        
#         # 计算最终的权重
#         weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype))
        
#         return weights.to(original_type)
    
class router_all_llama(nn.Module):
    def __init__(self, block, hidden_size):
        super().__init__()
        self.router_block = TokenRouter(hidden_size)  # 仅保留一个 Router
        self.block = block
        self.training_step = 0

        # 统计相关
        self.total_tokens = 0
        self.skipped_tokens = 0
        self.block_router_zero_prob = 0.0   

        # 存储 token 路由信息
        self.routing_matrix = {
            "block": None,
        }

        # 冻结底层 block 的所有参数（如果你想让其保持不训练的话）
        for param in self.block.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        b, s, _ = hidden_states.shape
        self.total_tokens += b * s

        # 防止出现 NaN
        if torch.isnan(hidden_states).any():
            warnings.warn("NaN detected in input tokens, please check your model.")

        # 防止 attention_mask 为 None
        if attention_mask is None:
            attention_mask_temp = torch.ones((b, s), device=hidden_states.device)
        else:
            attention_mask_temp = attention_mask
        temperature = 1.0  # 固定温度为 1.0，或者根据需要调整
        # ====== 核心：用一个 router 决定是否跳过整个 block ======
        # 1. 计算路由权重
        weights_block = self.router_block(hidden_states)  # (b, s, 2)

        # 2. Gumbel Softmax
        gumbel_weights_block = F.gumbel_softmax(weights_block, tau=temperature, hard=True, dim=-1)
        # gumbel_weights_block: (b, s, 2), 例如 [执行_prob, 跳过_prob] 或反过来

        # 这里假设维度 0 是「执行 gate」，维度 1 是「跳过 gate」，
        # 可以根据你自己的实现来取对应索引
        gate_execute = gumbel_weights_block[:, :, 0]  # 执行 gate
        gate_skip    = gumbel_weights_block[:, :, 1]  # 跳过 gate

        # 统计
        # 跳过 = 1 的地方，说明这个 token 直接用 residual
        self.skipped_tokens += gate_skip.sum().item()
        self.block_router_zero_prob = gate_execute.mean()  # 记录执行门的平均概率

        # 3. 执行或跳过 block
        residual = hidden_states

        # 3.1 如果执行，则真正算一次 block（包含 attention + mlp）
        #     这里是 LlamaBlock 的标准流程, 可能跟你的实现略有差异
        hidden_states_ln = self.block.input_layernorm(hidden_states)
        hidden_states_attn, self_attn_weights,present_key_value = self.block.self_attn(
            hidden_states=hidden_states_ln,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # post-attn
        hidden_states_attn_res = hidden_states_attn + residual
        # hidden_states_attn_res = hidden_states_attn_res * gate_execute.unsqueeze(-1) + residual * (1 - gate_execute).unsqueeze(-1)  # 如果 LLamaBlock 原本是这样加的

        # MLP
        hidden_states_ln_mlp = self.block.post_attention_layernorm(hidden_states_attn_res)
        hidden_states_mlp = self.block.mlp(hidden_states_ln_mlp)

        hidden_states_mlp = hidden_states_mlp + hidden_states_attn_res
        hidden_states = hidden_states_mlp * gate_execute.unsqueeze(-1) + residual * gate_skip.unsqueeze(-1)  # 同理，根据你的 LlamaBlock

        # # 3.2 用 Gumbel-softmax 的结果做混合：
        # #     - (执行) 就是 hidden_states_block
        # #     - (跳过) 就是 residual
        # # [b, s, hidden_dim]
        # hidden_states = hidden_states_block * gate_execute.unsqueeze(-1)  + residual* gate_skip.unsqueeze(-1)

        # 记录路由信息
        self.routing_matrix["block"] = gate_skip.to(torch.float32).detach().cpu().numpy()

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def compute_sparsity(self):
        skip_sparsity = self.skipped_tokens / self.total_tokens if self.total_tokens > 0 else 0
        return skip_sparsity

    def reset_sparsity_counts(self):
        self.total_tokens = 0
        self.skipped_tokens = 0


class router_all_gemma (nn.Module):
    def __init__(self, block, hidden_size, args):
        super().__init__()
        self.router_attention = TokenRouter(hidden_size)
        self.router_mlp = TokenRouter(hidden_size)
        self.block = block
        self.training_step = 0
        self.args= args

        # initialize the total tokens and skipped tokens
        self.total_tokens = 0
        self.skipped_attn_tokens = 0
        self.skipped_mlp_tokens = 0

        # record the sparsity of the routers
        self.attn_router_zero_prob = 0.0  
        self.mlp_router_zero_prob = 0.0   

        # 初始化存储 token 路由信息的字典
        self.routing_matrix = {
            "attention": None,
            "mlp": None
        }

        # freeze the parameters of the block
        for param in self.block.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # gemma特定代码
        if self.block.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.block.config._attn_implementation == "flash_attention_2":
                if past_key_value is not None:  # when decoding
                    attention_mask = attention_mask[:, -self.block.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.block.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.block.sliding_window :]
        b, s, _ = hidden_states.shape

        self.total_tokens += b * s

        # check for NaN in the input tokens
        if torch.isnan(hidden_states).any():
            warnings.warn(
                "NaN detected in input tokens, this is not intended to happen, please check your model.")

        # 防止attention mask为None
        if attention_mask is None:
            attention_mask_temp = torch.ones((b, s), device=hidden_states.device)
        else:
            attention_mask_temp = attention_mask

        temperature = 1.0

        # 计算gumbel softmax之前的权重
        weights = self.router_attention(hidden_states)

        # 计算gumbel softmax
        gumbel_weights = F.gumbel_softmax(weights, tau=temperature, hard=True, dim=-1)

        # gumbel weights的最后一个维度是长度为2的one-hot vectors，第一个代表是否执行，第二个代表是否跳过，我们取出第一个维度代表selected_mask
        selected_mask = gumbel_weights[:, :, 1]
        gumbel_weights_gate = gumbel_weights[:, :, 0]

        # 统计跳过 Attention 的次数
        self.skipped_attn_tokens += selected_mask.sum().item()
        # 记录router_attention的0类概率
        self.attn_router_zero_prob = gumbel_weights_gate.mean()

        # perform attention
        residual = hidden_states
        hidden_states = self.block.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.block.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = self.block.post_attention_layernorm(hidden_states)

         # 将attention的结果乘以gumbel weights
        hidden_states = hidden_states * gumbel_weights_gate.unsqueeze(-1) + residual
        
        # 计算mlp gumbel softmax之前的权重
        weights_mlp = self.router_mlp(residual)

        # 计算mlp的gumbel softmax
        gumbel_weights_mlp = F.gumbel_softmax(weights_mlp, tau=temperature, hard=True, dim=-1)

        # 计算gate
        selected_mask_mlp = gumbel_weights_mlp[:, :, 1]
        gumbel_weights_gate_mlp = gumbel_weights_mlp[:, :, 0]

        # 记录router_mlp的0类概率
        self.mlp_router_zero_prob = gumbel_weights_gate_mlp.mean()
        # 统计跳过 MLP 的次数
        self.skipped_mlp_tokens += selected_mask_mlp.sum().item()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.block.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        hidden_states = self.block.post_feedforward_layernorm(hidden_states)

        # # 将mlp的结果乘以gumbel weights
        hidden_states = hidden_states * gumbel_weights_gate_mlp.unsqueeze(-1) + residual
        # hidden_states = hidden_states  + residual

        # 记录最新的路由信息
        self.routing_matrix["attention"] = selected_mask.to(torch.float32).detach().cpu().numpy()
        self.routing_matrix["mlp"] = selected_mask_mlp.to(torch.float32).detach().cpu().numpy()
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def compute_sparsity(self):
        attn_sparsity = self.skipped_attn_tokens / self.total_tokens if self.total_tokens > 0 else 0
        mlp_sparsity = self.skipped_mlp_tokens / self.total_tokens if self.total_tokens > 0 else 0
        return attn_sparsity, mlp_sparsity

    def reset_sparsity_counts(self):
        self.total_tokens = 0
        self.skipped_attn_tokens = 0
        self.skipped_mlp_tokens = 0

def apply_router_all(model: PreTrainedModel) -> PreTrainedModel:
    hidden_size = model.config.hidden_size
    new_layers = nn.ModuleList()
    if model.__class__.__name__ == "LlamaForCausalLM":
        for i, layer in enumerate(model.model.layers):
            new_layer = router_all_llama(layer, hidden_size)
            new_layers.append(new_layer)
    
    elif model.__class__.__name__ == "Gemma2ForCausalLM":
        for i, layer in enumerate(model.model.layers):
            new_layer = router_all_gemma(layer, hidden_size)
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
