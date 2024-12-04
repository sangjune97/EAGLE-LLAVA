from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="cpu")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://i.namu.wiki/i/brFxhpvr8i82QGYJvQVOY-GJOR0n7ewuok48ldu8ZB1PxB5u0zkHAB6CdRxIIdMaifXRyFhz5aEt_NEhAa_nXsOiCc9fz-xuQUwx9tSPo8ej8q1BSU1m9qDpLdI1fAXHDxmK1ZDFLOsjxs2UdvV9Hw.webp"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    pixel_values=inputs["pixel_values"],
    max_new_tokens=64)

output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Outputs:\n")
print(output)