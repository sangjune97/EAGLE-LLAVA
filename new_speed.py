import json
import numpy as np
import os

def load_jsonl_field(jsonl_file, field_name):
    """JSONL 파일에서 특정 필드만 추출해서 리스트로 반환"""
    values = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            value = obj.get(field_name)
            if value is not None:
                values.append(value)
    return np.array(values)

def compare_jsonl_metrics_extended(file1, file2, fields_to_compare):
    """두 JSONL 파일에서 지정한 필드들의 평균 값을 비교 (항상 전체 경로 출력)"""
    print("\n📊 JSONL 파일 성능 비교")
    print(f"비교 대상 1: {file1}")
    print(f"비교 대상 2: {file2}\n")

    header_format = f"{'항목':<42} | {'파일1 평균':>15} | {'파일2 평균':>15} | {'비율 (file2/file1)':>20}"
    row_format = f"{'{:<42}'} | {'{:15.6f}'} | {'{:15.6f}'} | {'{:20.2f}'}"

    print(header_format)
    print("-" * len(header_format))

    for field in fields_to_compare:
        values1 = load_jsonl_field(file1, field)
        values2 = load_jsonl_field(file2, field)

        mean1 = values1.mean() if len(values1) > 0 else 0
        mean2 = values2.mean() if len(values2) > 0 else 0
        ratio = mean2 / mean1 if mean1 > 1e-8 else 0.0

        print(row_format.format(field, mean1, mean2, ratio))


if __name__ == "__main__":
    # 🔧 비교할 JSONL 파일 경로
    jsonl_file1 = "/data/sangjun/ckpt/token/finetune_w_img_1e-4_100/state_20/mmvet.jsonl"
    jsonl_file2 = "/data/sangjun/ckpt/token/finetune_w_img_1e-4_100_layer2/state_20/mmvet.jsonl"

    # 📌 비교할 항목 목록
    fields_to_compare = [
        "total_time",
        "tok_per_sec",
        "avg_accept_length",
        "initialize_time",
        "initialize_tree_time",
        "tree_decode_total_time",
        "evaluate_posterior_total_time",
        "update_inference_inputs_total_time"
    ]

    # 📈 비교 실행
    compare_jsonl_metrics_extended(jsonl_file1, jsonl_file2, fields_to_compare)
