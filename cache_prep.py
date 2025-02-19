import torch

from utils import (
    KVCache,
    Processor,
    convert_paper_to_text,
    get_position_ids,
    merge_kv_caches,
)

SYSTEM_TEMPLATE = "<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n<|im_start|>user\n["
SYSTEM_PROMPT = "You are a helpful data analyst. You will receive datas containing various fields and their corresponding values, representing different attributes. Use these fields to provide answers to the user query. The user query will indicate which fields to use for your response. Your response should contain only the answer and no additional formatting."


class QwenCachePrep(Processor):
    def __init__(self, model_name: str, device="auto"):
        """初始化QwenCachePrep类

        Args:
            model_name: Qwen模型的路径或名称
            device: 设备类型，默认为"auto"
        """
        super().__init__(model_name, device)
        system_prompt = SYSTEM_TEMPLATE.format(SYSTEM_PROMPT=SYSTEM_PROMPT)
        self.system_kv_cache = self._generate_cache(system_prompt)

    def get_kv_caches(self, contents: list[dict[str, str]]) -> list[KVCache]:
        """处理多个内容并生成token IDs和position IDs

        Args:
            contents: 包含多个内容的列表，每个内容是一个字典
        """
        current_position = self.system_kv_cache.length

        kv_caches = []
        current_kv_cache = None

        for idx, content in enumerate(contents):
            text = convert_paper_to_text(content)
            kv_cache = self._generate_cache(
                text,
                start_position=current_position,
                past_kv_cache=current_kv_cache,
            )
            if current_kv_cache is None:
                current_kv_cache = kv_cache
            else:
                current_kv_cache = merge_kv_caches([current_kv_cache, kv_cache])
            kv_caches.append(kv_cache)
            current_position += kv_cache.length

        return kv_caches

    def _generate_cache(
        self,
        text: str,
        start_position: int = 0,
        past_kv_cache: KVCache = None,
    ):
        """生成KV缓存对象"""
        token_ids = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False
        )
        token_length = token_ids.shape[-1]
        position_ids = get_position_ids(start_position, token_length)

        with torch.no_grad():
            output = self.model(
                input_ids=token_ids,
                position_ids=torch.tensor([position_ids], dtype=torch.long),
                use_cache=True,
                past_key_values=(
                    past_kv_cache.key_value_pairs if past_kv_cache else None
                ),
                output_hidden_states=True,
            )
        kv_cache = output.past_key_values
        return KVCache(kv_cache, token_length)
