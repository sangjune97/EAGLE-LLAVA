from eagle.model.ea_model import EaModel
import torch
from fastchat.model import get_conversation_template
model = EaModel.from_pretrained(
    base_model_path="meta-llama/Llama-2-7b-chat-hf",
    ea_model_path="yuhuili/EAGLE-llama2-chat-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=60
)
model.eval()
your_message="Hello"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])
print("Outputs:\n")
print(output)