import json
import numpy as np
import argparse
import os
import re

def load_jsonl_fields(jsonl_file, field_names):
    values_dict = {field: [] for field in field_names}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for field in field_names:
                value = obj.get(field)
                if value is not None:
                    values_dict[field].append(value)
                else:
                    values_dict[field].append(0)  # 없으면 0으로 대체
    return {field: np.array(vals) for field, vals in values_dict.items()}

def print_total_time_and_avg_accept_length(jsonl_file):
    filename = os.path.basename(jsonl_file)
    match = re.search(r'(\d+)\.jsonl$', filename)
    number = match.group(1) if match else ""

    fields = load_jsonl_fields(jsonl_file, ["encoding_time", "decoding_time", "avg_accept_length"])

    encoding = fields["encoding_time"]
    decoding = fields["decoding_time"]
    avg_accept = fields["avg_accept_length"]

    total_time = encoding + decoding

    total_time_mean = total_time.mean() if len(total_time) > 0 else 0
    avg_accept_length_mean = avg_accept.mean() if len(avg_accept) > 0 else 0

    print(number, total_time_mean, avg_accept_length_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL 파일의 total_time 및 avg_accept_length 계산")
    parser.add_argument(
        "jsonl_file", type=str,
        help="JSONL 파일 경로"
    )

    args = parser.parse_args()
    print_total_time_and_avg_accept_length(args.jsonl_file)
