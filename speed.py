import json
import numpy as np

# JSONL 파일 경로 설정
jsonl_file1 = "/home/sangjun/EAGLE-LLAVA/ckpt/finetune_pool_w_img_3e-5/state_40/test_pool_img.jsonl"  # 첫 번째 JSONL 파일 경로
jsonl_file2 = "/home/sangjun/LLaVA/scripts/v1_5/eval/playground/data/eval/mm-vet/answers_1gpu/llava-v1.5-7b.jsonl"  # 두 번째 JSONL 파일 경로

def calculate_avg_total_time(jsonl_file):
    """JSONL 파일에서 total_time 값을 읽어 평균을 계산하는 함수"""
    total_times = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            total_time = json_obj.get("total_time", 0)
            total_times.append(total_time)
    
    avg_time = np.mean(total_times)
    return avg_time

def calculate_avg_accept_length(jsonl_file):
    """JSONL 파일에서 avg_accept_length 값을 읽어 평균을 계산하는 함수"""
    accept_lengths = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            avg_accept_length = json_obj.get("avg_accept_length")
            if avg_accept_length is not None:
                accept_lengths.append(avg_accept_length)
    
    return np.mean(accept_lengths) if accept_lengths else 0

def calculate_avg_tok_per_sec(jsonl_file):
    """JSONL 파일에서 tok_per_sec 값을 읽어 평균을 계산하는 함수"""
    tok_per_secs = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            tok_per_sec = json_obj.get("tok_per_sec")
            if tok_per_sec is not None:
                tok_per_secs.append(tok_per_sec)
    
    return np.mean(tok_per_secs) if tok_per_secs else 0

# 평균 total_time 계산
avg_time1 = calculate_avg_total_time(jsonl_file1)
avg_time2 = calculate_avg_total_time(jsonl_file2)

# 첫 번째 파일의 avg_accept_length 계산
avg_accept_length1 = calculate_avg_accept_length(jsonl_file1)

# 각 파일의 tok_per_sec 평균 계산
avg_tok_per_sec1 = calculate_avg_tok_per_sec(jsonl_file1)
avg_tok_per_sec2 = calculate_avg_tok_per_sec(jsonl_file2)

print(jsonl_file1)
# 결과 출력
print(f"첫 번째 파일의 평균 total_time: {avg_time1:.6f} 초")
print(f"두 번째 파일의 평균 total_time: {avg_time2:.6f} 초")

# 두 파일의 속도 비교 (몇 배 더 빠른지)
if avg_time2 != 0:
    speed_ratio = avg_time2 / avg_time1
    print(f"첫 번째 파일이 두 번째 파일보다 {speed_ratio:.2f}배 더 빠릅니다.")
else:
    print("두 번째 파일의 평균 total_time이 0입니다. 비율을 계산할 수 없습니다.")

# tok_per_sec 비교
print(f"첫 번째 파일의 평균 tok_per_sec: {avg_tok_per_sec1:.6f}")
print(f"두 번째 파일의 평균 tok_per_sec: {avg_tok_per_sec2:.6f}")

if avg_tok_per_sec1 != 0:
    tok_speed_ratio = avg_tok_per_sec1 / avg_tok_per_sec2
    print(f"첫 번째 파일이 두 번째 파일보다 토큰 처리 속도가 {tok_speed_ratio:.2f}배 더 빠릅니다.")
else:
    print("첫 번째 파일의 평균 tok_per_sec가 0입니다. 비율을 계산할 수 없습니다.")

# 첫 번째 파일의 avg_accept_length 출력
print(f"첫 번째 파일의 평균 avg_accept_length: {avg_accept_length1:.6f}")
