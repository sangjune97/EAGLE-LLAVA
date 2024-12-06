from PIL import Image
import requests
import torch
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

# 모델 및 프로세서 초기화
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 초기 Prompt 및 이미지
prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://i.namu.wiki/i/brFxhpvr8i82QGYJvQVOY-GJOR0n7ewuok48ldu8ZB1PxB5u0zkHAB6CdRxIIdMaifXRyFhz5aEt_NEhAa_nXsOiCc9fz-xuQUwx9tSPo8ej8q1BSU1m9qDpLdI1fAXHDxmK1ZDFLOsjxs2UdvV9Hw.webp"
image = Image.open(requests.get(url, stream=True).raw)

# 입력 데이터 생성
inputs = processor(images=image, text=prompt, return_tensors="pt")

# 캐시 초기화
past_key_values = None
generated_text = ""

# 반복적으로 토큰 생성
for _ in range(6):  # 최대 64개의 토큰을 생성
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        past_key_values=past_key_values,
        use_cache=True,  # 캐시 사용 활성화
    )
    
    # 새로 생성된 토큰 ID와 캐시 추출
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
    past_key_values = outputs.past_key_values

    # 디코딩하여 텍스트 업데이트
    generated_text += tokenizer.decode(next_token_id, skip_special_tokens=False)

    # <EOS> 토큰이 생성되면 중단
    if tokenizer.eos_token_id in next_token_id:
        break

    # 다음 입력 업데이트
    inputs["input_ids"] = next_token_id.unsqueeze(0)
    print("input token is:",inputs["input_ids"])
    inputs["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=inputs["attention_mask"].device)],
        dim=1
    )
    print(inputs["attention_mask"].shape)

print("Generated text:", generated_text)