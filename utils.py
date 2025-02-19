import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def get_position_ids(start_position: int, length: int):
    """生成位置编码"""
    return list(range(start_position, start_position + length))


def convert_paper_to_text(content: dict[str, str], idx: int):
    """将论文转换为文本"""
    paper_id = f"paper{idx}"
    return json.dumps({paper_id: content}) + ", "


class KVCache:
    """KV缓存类，用于管理attention的key-value缓存及其位置编码"""

    def __init__(
        self,
        key_value_pairs: DynamicCache,
        length: int,
        name: str = None,
    ):
        """
        Args:
            key_value_pairs: 包含每一层transformer的key和value张量的列表
            position_ids: 对应的位置编码
        """
        self.key_value_pairs = key_value_pairs
        self.length = length
        self.name = name

    def save(self, cache_dir: str):
        """保存整个KVCache对象到文件"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = f"{cache_dir}/{self.name}_cache.pt"
        torch.save(self, cache_path)

    @classmethod
    def load(cls, cache_dir: str, name: str):
        """加载KVCache对象"""
        cache_path = f"{cache_dir}/{name}_cache.pt"
        return torch.load(cache_path)


class Processor:
    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer, self.model = self._init_model(model_name, device)
        self.system_kv_cache = None

    def _init_model(self, model_name: str, device: str):
        """初始化模型和tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        return tokenizer, model
