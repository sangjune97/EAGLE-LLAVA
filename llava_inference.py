from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat's the content of the image? explain in very detail. ASSISTANT:"
url = "https://i.namu.wiki/i/brFxhpvr8i82QGYJvQVOY-GJOR0n7ewuok48ldu8ZB1PxB5u0zkHAB6CdRxIIdMaifXRyFhz5aEt_NEhAa_nXsOiCc9fz-xuQUwx9tSPo8ej8q1BSU1m9qDpLdI1fAXHDxmK1ZDFLOsjxs2UdvV9Hw.webp"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="pt")
# Generate
generate_ids = model.generate(
    temperature=0.9,
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024)

output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
special_tokens = tokenizer.special_tokens_map  # 특수 토큰 이름 및 값
special_token_ids = {token: tokenizer.convert_tokens_to_ids(token_value) 
                     for token, token_value in special_tokens.items()}

print("특수 토큰과 해당 값:")
print(special_tokens)
print("\n특수 토큰 ID:")
print(special_token_ids)

print("Outputs:\n")
print(output)
print("Tokens:\n")
#print(generate_ids)