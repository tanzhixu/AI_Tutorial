import os
import json
import torch

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import Gemma_2bTransformerLayerWeight
from lightllm.models.gemma_2b.layer_weights.pre_and_post_layer_weight import Gemma_2bPreAndPostLayerWeight
from lightllm.models.gemma_2b.layer_infer.pre_layer_infer import Gemma_2bPreLayerInfer
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import Gemma_2bTransformerLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo

from lightllm.common.mem_utils import MemoryManager

class Gemma_2bTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = Gemma_2bPreAndPostLayerWeight
    transformer_weight_class = Gemma_2bTransformerLayerWeight

    # infer class
    pre_layer_infer_class = Gemma_2bPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Gemma_2bTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "gemma only supports HF format to load Now!"
        # assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _init_custom(self):
        self._init_to_get_rotary()
        return
    
    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(self.max_total_token_num,
                                        dtype=torch.float16,
                                        head_num=self.config["num_key_value_heads"], # [SYM] always == 1
                                        head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                        layer_num=self.config["num_hidden_layers"])       
        return


    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return
    