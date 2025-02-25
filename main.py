import argparse
import json
import os
import shutil
import time

from cache_prep import QwenCachePrep
from cache_query import KVCache, QwenQueryProcessor


def load_data(file_path):
    """加载数据集并提取需要的列"""
    data = json.load(open(file_path, "r"))
    return [{"title": d["title"], "abstract": d["abstract"]} for d in data]


def generate_cache(
    model_name: str,
    cache_dir: str,
    data_path: str,
    batch_size: int,
):
    """生成缓存文件"""
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

    print(f"Generating cache to {cache_dir}")
    for batch_idx in range(0, length, batch_size):
        batch_papers = papers[batch_idx : batch_idx + batch_size]
        kv_caches = processor.get_kv_caches(batch_papers)
        for i, kv_cache in enumerate(kv_caches):
            name = batch_idx + i
            kv_cache.save(cache_dir, name=f"{name}")
            print(f"Saved cache {name}, length: {kv_cache.length}")


def query_papers(
    model_name: str,
    cache_dir: str,
    batch_cache_dir: str,
    data_path: str,
    batch_size: int,
):
    """查询论文"""
    print(f"Querying papers using cache from {batch_cache_dir}")

    # 处理数据集
    papers = load_data(data_path)
    length = len(papers)
    print(f"Total records: {length}")
    length = length // batch_size * batch_size
    papers = papers[:length]
    # 加载缓存
    batch_kv_caches = [KVCache.load(batch_cache_dir, f"{i}") for i in range(length)]
    kv_caches = [KVCache.load(cache_dir, f"{i}") for i in range(length)]

    processor = QwenQueryProcessor(model_name, cache_dir)
    results = {}

    print("\nRunning queries...")
    results["no batch without cache"] = query_without_cache(processor, papers)
    print("Completed: no batch without cache")

    results["no batch with cache"] = query_with_cache(processor, kv_caches)
    print("Completed: no batch with cache")

    results["batch without cache"] = query_batch(processor, papers, batch_size)
    print("Completed: batch without cache")

    results["batch with cache"] = query_batch_with_cache(
        processor, batch_kv_caches, batch_size
    )
    print("Completed: batch with cache")

    # 打印结果
    print_results(results)


def query_without_cache(processor: QwenQueryProcessor, papers: list[dict]):
    """查询论文"""
    start = time.time()
    result = []
    for paper in papers:
        response = processor.query_without_cache(paper)
        print(response)
        result.extend(eval(response))

    return result, time.time() - start


def query_with_cache(processor: QwenQueryProcessor, kv_caches: list[KVCache]):
    """查询论文"""
    start = time.time()
    result = []
    for kv_cache in kv_caches:
        response = processor.query_with_cache(kv_cache)
        print(response)
        result.extend(eval(response))

    return result, time.time() - start


def query_batch(
    processor: QwenQueryProcessor, batch_papers: list[dict], batch_size: int
):
    """查询论文"""
    start = time.time()
    result = []
    for batch_idx in range(0, len(batch_papers), batch_size):
        batch_paper = batch_papers[batch_idx : batch_idx + batch_size]
        response = processor.query_batch_without_cache(batch_paper)
        print(response)
        result.extend(eval(response))

    return result, time.time() - start


def query_batch_with_cache(
    processor: QwenQueryProcessor, batch_kv_caches: list[KVCache], batch_size: int
):
    """查询论文"""
    start = time.time()
    result = []
    for batch_idx in range(0, len(batch_kv_caches), batch_size):
        batch_kv_cache = batch_kv_caches[batch_idx : batch_idx + batch_size]
        response = processor.query_batch_with_cache(batch_kv_cache)
        print(response)
        result.extend(eval(response))

    return result, time.time() - start


def calculate_accuracy(base_result: list, compare_result: list) -> float:
    """计算准确率"""
    if len(base_result) != len(compare_result):
        print(
            f"Warning: Length mismatch - base: {len(base_result)}, compare: {len(compare_result)}"
        )
        # 取最小长度进行比较
        length = min(len(base_result), len(compare_result))
        base_result = base_result[:length]
        compare_result = compare_result[:length]

    correct = sum(1 for b, c in zip(base_result, compare_result) if b == c)
    return (correct / len(base_result)) * 100


def print_results(results: dict):
    """打印结果"""
    print("\n" + "=" * 50)
    print("Results Summary:")
    print("=" * 50)

    # 获取基准结果
    base_result, base_time = results["no batch without cache"]
    print(f"\nBase Method (no batch without cache):")
    print(f"Time: {base_time:.2f}s")
    print(f"Accuracy: 100% (baseline)")

    # 比较其他方法
    methods = {
        "No Batch with Cache": "no batch with cache",
        "Batch without Cache": "batch without cache",
        "Batch with Cache": "batch with cache",
    }

    for method_name, key in methods.items():
        result, time_taken = results[key]
        accuracy = calculate_accuracy(base_result, result)

        print(f"\n{method_name}:")
        print(
            f"Time: {time_taken:.2f}s ({(time_taken/base_time*100):.1f}% of baseline)"
        )
        print(f"Accuracy: {accuracy:.2f}%")

    print("\n" + "=" * 50)


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
        "--batch-cache-dir",
        type=str,
        help="Directory for cache files",
        default="/fs/fast/share/pingtai_cc/cache-test/batch-cache",
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
    parser.add_argument(
        "--generate-batch-cache", action="store_true", help="Generate batch cache before querying"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        from cache_query import set_debug_mode

        set_debug_mode()

    if args.generate_cache:
        generate_cache(
            args.model_name,
            args.cache_dir,
            args.data_path,
            1,
        )
    if args.generate_batch_cache:
        generate_cache(
            args.model_name,
            args.batch_cache_dir,
            args.data_path,
            args.batch_size,
        )
    query_papers(
        args.model_name,
        args.cache_dir,
        args.batch_cache_dir,
        args.data_path,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
