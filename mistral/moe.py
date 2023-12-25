import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        # For each token, generate `num_experts` logits indicating which expert to use.
        gate_logits = self.gate(inputs)
        # For each token, select the top `num_experts_per_tok` experts, and use them to compute
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        # Apply the softmax to the logits AFTER selecting the top-k, this makes comparison with different hyperparams consitent.
        # Because even if we change the total number of experts or the number of experts per token, the sum of the weights will still be 1 for each token.
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for current_expert_index, current_expert in enumerate(self.experts):
            # For each expert, select which token it will be applied to.
            token_index, token_expert_index = torch.where(selected_experts == current_expert_index)
            # Apply the expert to the selected tokens weighting it by the logits (post-softmax) computed above .
            results[token_index] += weights[token_index, token_expert_index, None] * current_expert(
                inputs[token_index]
            )
        return results
