import torch
from transformers import DynamicCache

from cache_prep import SYSTEM_PROMPT
from utils import (
    KVCache,
    Processor,
    convert_paper_to_text,
    get_position_ids,
    merge_kv_caches,
)

QUERY_TEMPLATE = "{QUERY_PROMPT}\n<|im_end|>\n<|im_start|>assistant\n"
QUERY_PROMPT = [
    """]\nDetermine if each paper in the provided list is related to AI. The number of papers is {num_papers}.
For each paper, answer "yes" if the paper is about AI, and "no" if the paper is not about AI. 
Provide your answer in the following format: {example}, where each element corresponds to the respective paper.""",
    # """]\nTell me the json content of papers. Example: [{"title": "xxx", "abstract": "xxx"}, {"title": "xxx", "abstract": "xxx"}, {"title": "xxx", "abstract": "xxx"}, {"title": "xxx", "abstract": "xxx"}].""",
    """]\nTell me the number of papers. Example: 4.""",
]
MAX_TOKEN = 10000
DEBUG = False


def set_debug_mode():
    global DEBUG
    DEBUG = True


def _generate_example_answers(num_papers: int):
    """根据论文数量生成示例答案"""
    example = ["yes", "no"] * (num_papers // 2)
    if num_papers % 2:  # 如果是奇数，添加一个"yes"
        example.append("yes")
    return str(example)


def _generate_query_messages(num_papers: int):
    """根据论文数量生成查询消息"""
    if not DEBUG:
        example = _generate_example_answers(num_papers)
        return QUERY_PROMPT[0].format(num_papers=num_papers, example=example)
    return QUERY_PROMPT[1]


class QwenQueryProcessor(Processor):
    def __init__(self, model_name: str, cache_dir: str, device: str = "auto"):
        """初始化查询处理器

        Args:
            model_name: Qwen模型的路径或名称
            cache_dir: 缓存文件所在目录
            device: 设备类型，默认为"auto"
        """
        super().__init__(model_name, device)
        self.system_kv_cache = KVCache.load(cache_dir, "system")

    def query_without_cache(self, batch_papers: list[dict]):
        papers_content = "".join(convert_paper_to_text(paper) for paper in batch_papers)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"[{papers_content}{_generate_query_messages(len(batch_papers))}",
            },
        ]

        # 使用 tokenizer 的 chat template 处理消息
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # 生成响应
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKEN,
                num_beams=1,
                do_sample=False,  # 不使用采样
                temperature=1.0,  # 使用默认温度
                top_k=1,  # 只保留最可能的token
                top_p=1.0,  # 不使用nucleus sampling
                repetition_penalty=1.0,  # 不使用重复惩罚
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # 使用模型内部的KV缓存加速生成
            )

        # 只解码新生成的token
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response.strip()

    def query_with_cache(self, batch_kv_caches: list[KVCache]) -> str:
        merged_kv_cache = merge_kv_caches([self.system_kv_cache] + batch_kv_caches)

        # 处理查询``
        query_prompt = QUERY_TEMPLATE.format(
            QUERY_PROMPT=_generate_query_messages(len(batch_kv_caches))
        )
        query_token_ids = self.tokenizer.encode(
            query_prompt, return_tensors="pt", add_special_tokens=False
        )
        query_position_ids = get_position_ids(
            merged_kv_cache.length, query_token_ids.shape[-1]
        )

        return self._generate_response(
            query_token_ids, merged_kv_cache.key_value_pairs, query_position_ids
        )

    def _generate_response(
        self,
        input_ids: torch.Tensor,
        kv_cache: DynamicCache,
        position_ids: list[int],
    ) -> str:
        """生成响应"""
        input_ids = input_ids.to("cuda")
        position_ids = torch.tensor([position_ids], dtype=torch.long).to("cuda")
        output_ids = []

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )

            next_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1))
            if next_token_id == self.tokenizer.eos_token_id:
                return self.tokenizer.decode(output_ids)

            output_ids.append(next_token_id)
            past_key_values = outputs.past_key_values

            # 自回归生成时的position_id从上一个位置开始递增
            next_position = position_ids[0][-1].item() + 1

            for _ in range(MAX_TOKEN - 1):
                curr_position = torch.tensor([[next_position]], device="cuda")
                next_position += 1

                outputs = self.model(
                    input_ids=torch.tensor([[next_token_id]], device="cuda"),
                    position_ids=curr_position,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                next_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1))
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                output_ids.append(next_token_id)
                past_key_values = outputs.past_key_values

        response = self.tokenizer.decode(output_ids)
        return response.strip()
