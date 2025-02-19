import argparse
import json
import os
import shutil

from cache_prep import QwenCachePrep
from cache_query import KVCache, QwenQueryProcessor


def load_data(file_path):
    """加载数据集并提取需要的列"""
    data = json.load(open(file_path, "r"))
    return [{"title": d["title"], "abstract": d["abstract"]} for d in data]


def generate_cache(model_name: str, cache_dir: str, data_path: str, batch_size: int):
    """生成缓存文件"""
    print(f"Generating cache to {cache_dir}")

    # 清空或创建缓存目录
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # 处理数据集
    papers = load_data(data_path)
    length = len(papers)
    print(f"Total records: {length}")

    processor = QwenCachePrep(model_name)
    processor.system_kv_cache.save(cache_dir, name="system")

    for batch_idx in range(0, length, batch_size):
        batch_papers = papers[batch_idx : batch_idx + batch_size]
        kv_caches = processor.get_kv_caches(batch_papers)
        for i, kv_cache in enumerate(kv_caches):
            name = batch_idx + i
            kv_cache.save(cache_dir, name=f"{name}")
            print(f"Saved cache {name}, length: {kv_cache.length}")


def query_papers(model_name: str, cache_dir: str, data_path: str, batch_size: int):
    """查询论文"""
    print(f"Querying papers using cache from {cache_dir}")

    # 处理数据集
    papers = load_data(data_path)
    length = len(papers)
    print(f"Total records: {length}")
    length = length // batch_size * batch_size

    # 加载缓存
    kv_caches = [KVCache.load(cache_dir, f"{i}") for i in range(length)]

    processor = QwenQueryProcessor(model_name, cache_dir)
    same_count = 0
    total_count = 0

    for batch_idx in range(0, length, batch_size):
        batch_papers = papers[batch_idx : batch_idx + batch_size]
        batch_kv_caches = kv_caches[batch_idx : batch_idx + batch_size]

        response_without_cache = processor.query_without_cache(batch_papers)
        response_with_cache = processor.query_with_cache(batch_kv_caches)

        total_count += 1
        if response_without_cache == response_with_cache:
            same_count += 1

        print(
            f"Batch {batch_idx}:\nresponse_without_cache: {response_without_cache}\nresponse_with_cache: {response_with_cache}\n"
        )

    print(f"\nresult without cache vs with cache: {same_count / total_count: .2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache generation and query tool for paper analysis"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Path to the Qwen model",
        default="/fs/fast/u20247643/hf/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory for cache files",
        default="/fs/fast/share/pingtai_cc/cache-test/cache",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the input data file",
        default="/fs/fast/share/pingtai_cc/prompt-cache-test1/arxiv-metadata-oai-snapshot-sample.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for processing (default: 2)",
    )
    parser.add_argument(
        "--generate-cache", action="store_true", help="Generate cache before querying"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        from cache_query import set_debug_mode

        set_debug_mode()

    if args.generate_cache:
        generate_cache(args.model_name, args.cache_dir, args.data_path, args.batch_size)
    query_papers(args.model_name, args.cache_dir, args.data_path, args.batch_size)


if __name__ == "__main__":
    main()
