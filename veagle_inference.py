from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from eagle.model.ea_model import EaModel
import torch
import os
from fastchat.model import get_conversation_template

# GPU 인덱스 0과 1을 사용하도록 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
processor = AutoProcessor.from_pretrained("/home/sangjun/llava-1.5-7b-hf-cls")

model = EaModel.from_pretrained(
    base_model_path="/home/sangjun/llava-1.5-7b-hf-cls",
    ea_model_path="/home/sangjun/EAGLE-LLAVA/ckpt/w_img_5e-5/state_40",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=60,
    temperature=0,
)
#yuhuili/EAGLE-Vicuna-13B-v1.3
model.eval()

prompt = "USER: <image>\nWhat is the image? ASSISTANT:"
url = "https://www.pncc.govt.nz/files/assets/public/v/2/images/services/animals/doggos.jpg?w=1920&h=1080"
image = Image.open(requests.get(url, stream=True).raw)
#image = Image.open(os.path.join("/home/sangjun/LLaVA/playground/data/eval/mm-vet/images/v1_30.jpg")).convert('RGB')
#image = Image.open(os.path.join("/data/COCO/train2017/000000027989.jpg")).convert('RGB')

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
generate_ids, new_token, idx, avg_accept_length, colorized_text  = model.eagenerate(
    input_ids = torch.as_tensor(inputs["input_ids"]).cuda(),
    temperature=0,
    log=True,
    pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda(),
    max_new_tokens=1024,
    token_process=0,
    color=True
    )
#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_except_first
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
print("Outputs:\n")
print(output)
print(avg_accept_length)
print(colorized_text)
import pdb;pdb.set_trace()








#########################################################################################################################

# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# # (1) 예시 어텐션 맵 & input_ids 준비
# attention_map = attentions[0]   # torch.Size([1, 604, 604])
# input_ids = torch.as_tensor(inputs["input_ids"])  # torch.Size([1, 604]) 예시

# # (2) 배치 차원 제거
# attention_map = attention_map.squeeze(0)  # -> (604, 604)
# import torch.nn.functional as F

# kerne_size = 6
# attention_map = F.avg_pool2d(attention_map.unsqueeze(0).unsqueeze(0), kernel_size=kerne_size)
# # 평균 풀링 후 (1, 1, 302, 302) 형태이므로 squeeze()를 통해 최종 (302, 302)로 만듭니다.
# attention_map = attention_map.squeeze(0).squeeze(0)
# #import pdb;pdb.set_trace()
# input_ids = input_ids.squeeze(0)          # -> (604,)

# # (3) input_ids -> 토큰 문자열로 변환
# tokenzier = processor.tokenizer

# tokens = tokenzier.decode(input_ids)

# first_index = next(i for i, token in enumerate(input_ids) if token == 32000)
# positions_to_label = list(range(first_index, min(first_index + 576, len(input_ids))))
# labels = ["img" for _ in positions_to_label]
# downsampled_positions = [pos // kerne_size for pos in positions_to_label]

# # (4) 어텐션 맵 시각화
# plt.figure(figsize=(16, 16))

# vmin = 1e-4
# vmax = 1
# attention_map[attention_map <= 0] = vmin
# plt.imshow(attention_map.cpu().float(), cmap='viridis', norm = LogNorm(vmin=vmin,vmax=vmax),aspect='auto')

# # (5) 축 라벨 지정
# #plt.xticks(downsampled_positions, labels, rotation=90, fontsize=6)
# #plt.yticks(downsampled_positions, labels, fontsize=6)

# # 틱 선 제거
# plt.tick_params(axis='x', which='both', length=0)
# plt.tick_params(axis='y', which='both', length=0)

# plt.colorbar()
# plt.title('Attention Score Map')
# plt.tight_layout()  # 라벨 겹침 방지
# plt.show()
# plt.savefig('attention_map.png', dpi=600, bbox_inches='tight')





#########################With Image###############################
# token_nums = -25
# fontsize = 8
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# # (1) 예시 어텐션 맵 & input_ids 준비
# attention_map = attentions[0]   # torch.Size([1, 604, 604])
# input_ids = torch.as_tensor(inputs["input_ids"])  # torch.Size([1, 604]) 예시

# # (2) 배치 차원 제거
# attention_map = attention_map.squeeze(0)  # -> (604, 604)
# import torch.nn.functional as F
# #import pdb;pdb.set_trace()
# input_ids = input_ids.squeeze(0)          # -> (604,)

# # (3) input_ids -> 토큰 문자열로 변환
# tokenzier = processor.tokenizer

# tokens = tokenzier.convert_ids_to_tokens(input_ids)[token_nums:]
# # (4) 어텐션 맵 시각화
# plt.figure(figsize=(6,6))

# vmin = 1e-4
# vmax = 1
# attention_map[attention_map <= 0] = vmin
# plt.imshow(attention_map[token_nums:,token_nums:].cpu().float(), cmap='viridis', norm = LogNorm(vmin=vmin,vmax=vmax),aspect='auto')

# # (5) 축 라벨 지정
# plt.xticks(range(abs(token_nums)), tokens, rotation=90, fontsize=fontsize)
# plt.yticks(range(abs(token_nums)), tokens, fontsize=fontsize)

# # 틱 선 제거
# plt.tick_params(axis='x', which='both', length=0)
# plt.tick_params(axis='y', which='both', length=0)

# plt.colorbar()
# plt.title('Attention Score Map')
# plt.tight_layout()  # 라벨 겹침 방지
# plt.show()
# plt.savefig('attention_map.png', dpi=300, bbox_inches='tight')

########################Without Image###############################
# def remove_image_token(input_ids, img_tok_index, hidden_states=None):
#     mask = input_ids != img_tok_index
#     filtered_input_ids = input_ids[mask].view(1, -1).to(input_ids.device)
#     if hidden_states is not None:
#         filtered_hidden_states = hidden_states[:, mask[0], :]
#         return filtered_input_ids, filtered_hidden_states
    
#     return filtered_input_ids

# fontsize = 8
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# # (1) 예시 어텐션 맵 & input_ids 준비
# attention_map = attentions[0]   # torch.Size([1, 604, 604])
# print(torch.as_tensor(inputs["input_ids"]).shape)
# input_ids = remove_image_token(torch.as_tensor(inputs["input_ids"]),32000)  # torch.Size([1, 604]) 예시
# print(input_ids.shape)
# # (2) 배치 차원 제거
# attention_map = attention_map.squeeze(0)  # -> (604, 604)
# import torch.nn.functional as F
# #import pdb;pdb.set_trace()
# input_ids = input_ids.squeeze(0)          # -> (604,)

# # (3) input_ids -> 토큰 문자열로 변환
# tokenzier = processor.tokenizer

# tokens = tokenzier.convert_ids_to_tokens(input_ids)

# # (4) 어텐션 맵 시각화
# plt.figure(figsize=(6,6))

# vmin = 1e-3
# vmax = 1
# attention_map[attention_map <= 0] = vmin
# plt.imshow(attention_map.cpu().float(), cmap='viridis', norm = LogNorm(vmin=vmin,vmax=vmax),aspect='auto')

# # (5) 축 라벨 지정
# plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=fontsize)
# plt.yticks(range(len(tokens)), tokens, fontsize=fontsize)

# # 틱 선 제거
# plt.tick_params(axis='x', which='both', length=0)
# plt.tick_params(axis='y', which='both', length=0)

# plt.colorbar()
# plt.title('Attention Score Map')
# plt.tight_layout()  # 라벨 겹침 방지
# plt.show()
# plt.savefig('attention_map.png', dpi=300, bbox_inches='tight')