import json
import numpy as np
import argparse

def load_jsonl_field(jsonl_file, field_name):
    values = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            value = obj.get(field_name)
            if value is not None:
                values.append(value)
    return np.array(values)

def compare_jsonl_metrics_extended(file1, file2, fields_to_compare):
    print("\n📊 JSONL 파일 성능 비교")
    print(f"비교 대상 1: {file1}")
    print(f"비교 대상 2: {file2}\n")

    header_format = "{:<42} | {:>15} | {:>15} | {:>20}"
    row_format    = "{:<42} | {:15.6f} | {:15.6f} | {:20.2f}"

    print(header_format.format("항목", "파일1 평균", "파일2 평균", "비율 (file2/file1)"))
    print("-" * 95)

    for field in fields_to_compare:
        values1 = load_jsonl_field(file1, field)
        values2 = load_jsonl_field(file2, field)

        mean1 = values1.mean() if len(values1) > 0 else 0.0
        mean2 = values2.mean() if len(values2) > 0 else 0.0
        ratio = mean2 / mean1 if mean1 > 1e-8 else 0.0

        print(row_format.format(field, mean1, mean2, ratio))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="두 JSONL 파일의 필드별 평균 비교")
    parser.add_argument(
        "--jsonl_file1", type=str,
        default="/data/sangjun/ckpt/token/cls/layer1/20_sharegpt/state_20/mmvet10.jsonl",
        help="비교 대상 파일 1 경로"
    )
    parser.add_argument(
        "--jsonl_file2", type=str,
        default="/home/sangjun/LLaVA/playground/data/eval/mm-vet/answers_1gpu/llava-v1.5-7b.jsonl",
        help="비교 대상 파일 2 경로"
    )

    args = parser.parse_args()

    fields_to_compare = [
        "encoding_time",
        "decoding_time",
        "tok_per_sec",
        "avg_accept_length",
        "initialize_time",
        "initialize_tree_time",
        "tree_decode_total_time",
        "evaluate_posterior_total_time",
        "update_inference_inputs_total_time",
        "num_tokens"       # ← 여기 추가!
    ]

    compare_jsonl_metrics_extended(args.jsonl_file1, args.jsonl_file2, fields_to_compare)
