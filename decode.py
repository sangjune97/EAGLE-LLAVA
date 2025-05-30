from transformers import AutoTokenizer
import torch

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0  # 혹시 없으면 0으로

# 원본 input_ids 리스트들 (길이 다름)
raw_input_ids = [
    [319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 1724, 2411, 5795, 947, 278, 7945, 29915, 29879, 4423, 505, 363, 278, 4272, 29915, 29879, 8608, 362, 1788, 29973, 13] + [32000]*100,
    [319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871] + [32000]*100 + [29871, 13, 5618, 338, 263, 7037, 2769, 363, 4856, 304, 6755, 445, 1134, 310, 7375, 11203],
    [319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871] + [32000]*100 + [29871, 13, 5618, 29915, 29879, 10464, 297, 278, 9088, 29973, 319, 1799, 9047, 13566, 29901, 450],
    [319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 3750, 1795, 278, 11203, 367, 289, 935, 292, 2978, 278, 11952, 29973, 13] + [32000]*100 + [29871, 319, 1799, 9047],
]

# 가장 긴 시퀀스 길이로 패딩
max_len = max(len(seq) for seq in raw_input_ids)
padded_input_ids = [seq + [pad_id] * (max_len - len(seq)) for seq in raw_input_ids]

# 텐서 변환
input_ids = torch.tensor(padded_input_ids, dtype=torch.long)

# 디코딩
decoded = [tokenizer.decode(ids, skip_special_tokens=False) for ids in input_ids]

# 출력
for i, text in enumerate(decoded):
    print(f"[Sample {i}]\n{text}\n{'='*80}")