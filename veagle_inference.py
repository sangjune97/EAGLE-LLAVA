from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from eagle.model.ea_model import EaModel
import torch
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

model = EaModel.from_pretrained(
    base_model_path="llava-hf/llava-1.5-7b-hf",
    ea_model_path="yuhuili/EAGLE-Vicuna-7B-v1.3",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://i.namu.wiki/i/brFxhpvr8i82QGYJvQVOY-GJOR0n7ewuok48ldu8ZB1PxB5u0zkHAB6CdRxIIdMaifXRyFhz5aEt_NEhAa_nXsOiCc9fz-xuQUwx9tSPo8ej8q1BSU1m9qDpLdI1fAXHDxmK1ZDFLOsjxs2UdvV9Hw.webp"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="pt")

# Generate
generate_ids = model.eagenerate(
    input_ids=torch.as_tensor(inputs["input_ids"]).cuda(), 
    attention_mask=torch.as_tensor(inputs["attention_mask"]).cuda(), 
    pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda(),
    max_new_tokens=256)

output = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
print("Outputs:\n")
print(output)