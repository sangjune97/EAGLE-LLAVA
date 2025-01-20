from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from eagle.model.ea_model import EaModel
import torch
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

model = EaModel.from_pretrained(
    base_model_path="llava-hf/llava-1.5-7b-hf",
    ea_model_path="/home/sangjun/EAGLE-LLAVA/ckpt/state_40",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1,
)
#yuhuili/EAGLE-Vicuna-13B-v1.3
model.eval()

prompt = "USER: <image>\nWhat's the content of the image? explain in very detail. ASSISTANT:"
url = "https://media.istockphoto.com/id/92450205/photo/sunsplashed-window.jpg?s=612x612&w=0&k=20&c=dTuhETbiWnoxAR1Ek5ROlj0liKxBazb14d9rsfe4XTc="
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="pt")

# Generate

generate_ids, new_token, idx, avg_accept_length  = model.eagenerate(
    temperature=0,
    log=True,
    input_ids=torch.as_tensor(inputs["input_ids"]).cuda(), 
    attention_mask=torch.as_tensor(inputs["attention_mask"]).cuda(), 
    pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda(),
    max_new_tokens=1024)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
print("Outputs:\n")
print(output)
print(avg_accept_length)