import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_position_ids(start_position: int, length: int):
    """生成位置编码"""
    return list(range(start_position, start_position + length))


def convert_paper_to_text(content: dict[str, str]):
    """将论文转换为文本"""
    return json.dumps(content) + ", "


class KVCache:
    """KV缓存类，用于管理attention的key-value缓存及其位置编码"""

    def __init__(
        self,
        key_value_pairs: DynamicCache,
        length: int,
    ):
        """
        Args:
            key_value_pairs: 包含每一层transformer的key和value张量的列表
            position_ids: 对应的位置编码
        """
        self.key_value_pairs = key_value_pairs
        self.length = length

    def save(self, cache_dir: str, name: str):
        """保存整个KVCache对象到文件"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = f"{cache_dir}/{name}_cache.pt"
        torch.save(self, cache_path)

    @classmethod
    def load(cls, cache_dir: str, name: str) -> "KVCache":
        """加载KVCache对象"""
        cache_path = f"{cache_dir}/{name}_cache.pt"
        return torch.load(cache_path)


class Processor:
    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer, self.model = self._init_model(model_name, device)
        self.system_kv_cache = None

    def _init_model(
        self, model_name: str, device: str
    ) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
        """初始化模型和tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        return tokenizer, model


def merge_kv_caches(kv_caches: list[KVCache]):
    """合并多个KV缓存"""
    merged_kv_cache = []

    # 对每一层transformer进行合并
    num_layers = len(kv_caches[0].key_value_pairs)
    for layer_idx in range(num_layers):
        layer_keys = [cache.key_value_pairs.key_cache[layer_idx] for cache in kv_caches]
        layer_values = [
            cache.key_value_pairs.value_cache[layer_idx] for cache in kv_caches
        ]

        merged_layer_k = torch.cat(layer_keys, dim=2)
        merged_layer_v = torch.cat(layer_values, dim=2)

        merged_kv_cache.append((merged_layer_k, merged_layer_v))

    merged_length = sum(cache.length for cache in kv_caches)

    return KVCache(DynamicCache.from_legacy_cache(merged_kv_cache), merged_length)
