from PIL import Image
import requests
import inspect
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained("/home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369", device_map="auto")
processor = AutoProcessor.from_pretrained("/home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369")
tokenizer = AutoTokenizer.from_pretrained("/home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369")

prompt = "USER: <image>\nWhat's the content of the image? explain in very detail. ASSISTANT:"
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKhJ9vY-WJviH34cgDfbG2Hn_cBf0t5BBmaWrmH--NzBO3pjGP6hjV7pb8s958ug9K7p6iR-3vz6nlw7c4i5ZdMw"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="pt")
print(inputs["input_ids"])
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