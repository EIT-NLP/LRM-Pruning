# inspired by  https://github.com/kyegomez/Mixture-of-Depths
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

from transformers import PreTrainedModel, DynamicCache, Cache
# class TokenRouter(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         # 直接从输入维度到输出权重预测
#         self.weight_predictor = nn.Linear(embed_dim, 2)
        
#         # 使用 He Kaiming 初始化
#         nn.init.kaiming_uniform_(self.weight_predictor.weight, nonlinearity='linear')
        
#         # 初始化 bias 为 0
#         if self.weight_predictor.bias is not None:
#             nn.init.zeros_(self.weight_predictor.bias)

#     def forward(self, x):
#         # 保存输入的原始数据类型
#         original_type = x.dtype
        
#         # 计算权重预测
#         weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype))
        
#         return weights.to(original_type)

class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 直接从输入维度到输出权重预测
        self.weight_predictor = nn.Linear(embed_dim, 2)
        # self.weight_predictor = nn.Linear(embed_dim, 1)
        
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

"""
class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 中间层的维度是 embed_dim 的四分之一
        intermediate_dim = embed_dim //4
        # 增加一个中间层
        self.hidden_layer = nn.Linear(embed_dim, intermediate_dim)
        self.relu = nn.ReLU() 
        self.weight_predictor = nn.Linear(intermediate_dim, 2)
        
        # 使用 He Kaiming 初始化
        nn.init.kaiming_uniform_(self.hidden_layer.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight_predictor.weight, nonlinearity='linear')

        # 初始化 bias 为 0
        if self.hidden_layer.bias is not None:
            nn.init.zeros_(self.hidden_layer.bias)
        if self.weight_predictor.bias is not None:
            nn.init.zeros_(self.weight_predictor.bias)

    def forward(self, x):
        original_type = x.dtype
        
        # 先通过中间层并激活，再传递到 weight_predictor
        x = self.hidden_layer(x.to(self.hidden_layer.weight.dtype))
        x = self.relu(x)  # 使用 ReLU 激活函数
        
        # 计算最终的权重
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype))
        
        return weights.to(original_type)
"""
class router_attn_mlp_llama (nn.Module): # new_layer = router_attn_mlp_llama(layer, hidden_size, args)
    def __init__(self, block, hidden_size):
        super().__init__()
        self.router_attention = TokenRouter(hidden_size)
        self.router_mlp = TokenRouter(hidden_size)
        self.block = block
        self.training_step = 0

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

    def forward( # new_layer = router_attn_mlp_llama(layer, hidden_size, args)
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        b, s, _ = hidden_states.shape

        # check for NaN in the input tokens
        if torch.isnan(hidden_states).any():
            warnings.warn(
                "NaN detected in input tokens, this is not intended to happen, please check your model.")
    
        # 防止attention mask为None
        if attention_mask is None:
            attention_mask = torch.ones((b, s), device=hidden_states.device)
        # 更新 self.total_tokens，只统计 attention_mask 为1的 token
        self.total_tokens += attention_mask.sum().item()

        
        temperature = 1.0
        use_gumble_softmax = True
        # use_cache = False
        if use_gumble_softmax:
            # 计算gumbel softmax之前的权重
            weights = self.router_attention(hidden_states) 

            # 计算gumbel softmax
            gumbel_weights = F.gumbel_softmax(weights, tau=temperature, hard=True, dim=-1)
            # gumbel weights的最后一个维度是长度为2的one-hot vectors，第一个代表是否执行，第二个代表是否跳过，我们取出第一个维度代表selected_mask
            selected_mask = gumbel_weights[:, :, 1] * attention_mask 
            gumbel_weights_gate = gumbel_weights[:, :, 0]

            # 统计跳过 Attention 的次数
            self.skipped_attn_tokens += selected_mask.sum().item()
            # 记录router_attention的0类概率
            self.attn_router_zero_prob = gumbel_weights_gate.mean()
            # import ipdb; ipdb.set_trace()
            # perform attention
            residual = hidden_states
            hidden_states = self.block.input_layernorm(hidden_states)   
            hidden_states, self_attn_weights = self.block.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            # 将attention的结果乘以gumbel weights
            hidden_states = hidden_states * gumbel_weights_gate.unsqueeze(-1) + residual

            # #新公式
            # hidden_states = (hidden_states + residual)*gumbel_weights_gate.unsqueeze(-1) + residual*selected_mask.unsqueeze(-1)
            
            # 计算mlp gumbel softmax之前的权重
            weights_mlp = self.router_mlp(residual)

            # 计算mlp的gumbel softmax
            gumbel_weights_mlp = F.gumbel_softmax(weights_mlp, tau=temperature, hard=True, dim=-1)

            # 计算gate
            selected_mask_mlp = gumbel_weights_mlp[:, :, 1] * attention_mask 
            gumbel_weights_gate_mlp = gumbel_weights_mlp[:, :, 0]

            # 记录router_mlp的0类概率
            self.mlp_router_zero_prob = gumbel_weights_gate_mlp.mean()
            # 统计跳过 MLP 的次数
            self.skipped_mlp_tokens += selected_mask_mlp.sum().item()

            # Fully Connected
            residual = hidden_states
            hidden_states = self.block.post_attention_layernorm(hidden_states)
            hidden_states = self.block.mlp(hidden_states)

            # # 将mlp的结果乘以gumbel weights
            hidden_states = hidden_states * gumbel_weights_gate_mlp.unsqueeze(-1) + residual
            # hidden_states = hidden_states  + residual

            # #新公式mlp routing
            # hidden_states = (hidden_states + residual)*gumbel_weights_gate_mlp.unsqueeze(-1) + residual*selected_mask_mlp.unsqueeze(-1)
    
            # 记录最新的路由信息
            self.routing_matrix["attention"] = selected_mask.to(torch.float32).detach().cpu().numpy()
            self.routing_matrix["mlp"] = selected_mask_mlp.to(torch.float32).detach().cpu().numpy()
            
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self.block.self_attn,)

            # if use_cache:
            #     outputs += (present_key_value,)

            # if output_attentions:
            #     outputs += (self_attn_weights,)

            return outputs
        else: # 4096 --> 1, use sigmoid and STE
            # 计算 Sigmoid and STE 之前的权重
            weights = self.router_attention(hidden_states) 

            # weights shape: [B, S_l, 1]
            sigmoid_weights = gumbel_sigmoid(weights)
            selected_mask = sigmoid_weights.squeeze(-1) * attention_mask  # shape: [B, S_l]
            sigmoid_weights_gate = 1.0 - sigmoid_weights.squeeze(-1)
            # 统计跳过 Attention 的次数
            self.skipped_attn_tokens += selected_mask.sum().item()
            # 记录router_attention的0类概率
            self.attn_router_zero_prob = sigmoid_weights_gate.mean() # 为什么不使用 item() ?
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
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            # 将attention的结果乘以 sigmoid weights
            hidden_states = hidden_states * sigmoid_weights_gate.unsqueeze(-1) + residual

            # #新公式
            # hidden_states = (hidden_states + residual)*sigmoid_weights_gate.unsqueeze(-1) + residual*selected_mask.unsqueeze(-1)
            
            # 计算mlp sigmoid softmax之前的权重
            weights_mlp = self.router_mlp(residual)

            # 计算 mlp 的 gumbel_sigmoid
            sigmoid_weights_mlp = gumbel_sigmoid(weights_mlp)

            # 计算gate
            selected_mask_mlp = sigmoid_weights_mlp.squeeze(-1) * attention_mask  
            sigmoid_weights_gate_mlp = 1.0 - sigmoid_weights_mlp.squeeze(-1)

            # 记录router_mlp的0类概率
            self.mlp_router_zero_prob = sigmoid_weights_gate_mlp.mean()
            # 统计跳过 MLP 的次数
            self.skipped_mlp_tokens += selected_mask_mlp.sum().item()

            # Fully Connected
            residual = hidden_states
            hidden_states = self.block.post_attention_layernorm(hidden_states)
            hidden_states = self.block.mlp(hidden_states)

            # # 将mlp的结果乘以 sigmoid weights
            hidden_states = hidden_states * sigmoid_weights_gate_mlp.unsqueeze(-1) + residual
            # hidden_states = hidden_states  + residual

            # #新公式mlp routing
            # hidden_states = (hidden_states + residual)*sigmoid_weights_gate_mlp.unsqueeze(-1) + residual*selected_mask_mlp.unsqueeze(-1)
    
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

class router_attn_mlp_gemma (nn.Module):
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
            attention_mask = torch.ones((b, s), device=hidden_states.device)

        # 训练过程中temperature逐渐降低
        if self.router_attention.training:
            if self.training_step <  self.args.gradient_accumulation_steps * self.args.max_steps_stage:
                self.training_step += 1
            temperature = self.args.initial_temperature - (self.args.initial_temperature - self.args.final_temperature) * ((self.training_step-1) // self.args.gradient_accumulation_steps )/ ( self.args.max_steps_stage)
        else:
            temperature = self.args.final_temperature

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

def apply_router_attn_mlp(model: PreTrainedModel) -> PreTrainedModel:
    hidden_size = model.config.hidden_size
    new_layers = nn.ModuleList()

    if model.__class__.__name__ == "LlamaForCausalLM":
        for i, layer in enumerate(model.model.layers):
            new_layer = router_attn_mlp_llama(layer, hidden_size)
            new_layers.append(new_layer)
    
    elif model.__class__.__name__ == "Gemma2ForCausalLM":
        for i, layer in enumerate(model.model.layers):
            new_layer = router_attn_mlp_gemma(layer, hidden_size)
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


def sigmoid_ste(logits):
    """
    Sigmoid + Straight-Through Estimator 二值离散采样
    :param logits: 输入的logits张量
    :return: 二值化结果，前向离散，后向可导
    """
    probs = torch.sigmoid(logits)                   # 概率
    # binary_sample = (probs > 0.5).float()           # 前向离散
    binary_sample = (probs > 0.5).to(probs.dtype) # 会导致推理问题
    output = binary_sample + probs - probs.detach() # 后向STE补梯度
    return output
def sample_gumbel(shape, eps=1e-10, device='cpu'):
    """Sample from Gumbel(0,1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid(logits, tau=1.0, hard=False):
    """
    Gumbel-Sigmoid: Differentiable Bernoulli (0/1) sampling

    Args:
        logits: Tensor of shape [*,] — input logits
        tau: Temperature parameter (lower = closer to binary)
        hard: If True, output hard 0/1 during forward, keep soft gradient

    Returns:
        Tensor of shape [*,] — sampled values in [0,1] (soft), or {0,1} (hard)
    """
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau).to(logits.dtype)  # Ensure the output is the same dtype as logits

    if hard:
        y_hard = (y_soft > 0.5).float()
        # Straight-through estimator
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft