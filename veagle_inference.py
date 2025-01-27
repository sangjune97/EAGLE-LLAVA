from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from eagle.model.ea_model import EaModel
import torch
import os
from fastchat.model import get_conversation_template

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

model = EaModel.from_pretrained(
    base_model_path="llava-hf/llava-1.5-7b-hf",
    ea_model_path="/home/sangjun/EAGLE-LLAVA/ckpt/new/state_50",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=60,
)
#yuhuili/EAGLE-Vicuna-13B-v1.3
model.eval()

prompt = "USER: <image>\nWhat type of elephant is being transported on the truck? and what is your name? What can you do for me? ASSISTANT:"
#url = "https://media.istockphoto.com/id/92450205/photo/sunsplashed-window.jpg?s=612x612&w=0&k=20&c=dTuhETbiWnoxAR1Ek5ROlj0liKxBazb14d9rsfe4XTc="
#image = Image.open(requests.get(url, stream=True).raw)
#image = Image.open(os.path.join("/home/sangjun/LLaVA/playground/data/eval/mm-vet/images/v1_30.jpg")).convert('RGB')
image = Image.open(os.path.join("/data/COCO/train2017/000000027989.jpg")).convert('RGB')

inputs = processor(images=image, text=prompt, return_tensors="pt")

#your_message="Do you know what is the purpose of life?"
#conv = get_conversation_template("vicuna")
#conv.append_message(conv.roles[0], your_message)
#conv.append_message(conv.roles[1], None)
#prompt = conv.get_prompt()
#tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.3')
#input_ids=tokenizer([prompt]).input_ids
#input_ids = torch.as_tensor(input_ids).cuda()
#
generate_ids, new_token, idx, avg_accept_length  = model.eagenerate(
    input_ids = torch.as_tensor(inputs["input_ids"]).cuda(),
    temperature=0,
    log=True,
    pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda(),
    max_new_tokens=1024,
    )
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
print("Outputs:\n")
print(output)
print(avg_accept_length)
