import argparse
import gc
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template
from PIL import Image

bigname="/home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369"
#bigname="lmsys/vicuna-7b-v1.5"
        
def keep_topk_image_token(
    input_ids,
    loss_mask,
    hidden_states,
    image_features,
    attentions,
    img_tok_index=32000,
    topk=50,
):
    """
    input_ids: [1, seq_len]
    loss_mask: [1, seq_len]
    hidden_states: [1, seq_len, dim]
    attentions: list of [1, heads, seq_len, seq_len]
    image_features: [1, 576, dim] or [576, dim]
    """
    # 기준 디바이스 정하기 (input_ids가 기준)
    device = input_ids.device

    # 텐서 차원 축소 및 디바이스 정렬
    input_ids = input_ids[0].to(device)        # [seq_len]
    loss_mask = loss_mask[0].to(device)        # [seq_len]
    hidden_states = hidden_states[0].to(device)  # [seq_len, dim]

    # 마지막 토큰이 attend한 attention 값 추출 후 평균
    last_layer_attn = attentions[-1][0].to(device)         # [heads, seq_len, seq_len]
    last_token_attention = last_layer_attn[:, -1, :].mean(dim=0)  # [seq_len]

    # 이미지 토큰 인덱스 추출
    image_token_indices = (input_ids == img_tok_index).nonzero(as_tuple=True)[0]

    # 예외 처리: 이미지 토큰이 없는 경우
    if image_token_indices.size(0) == 0:
        return input_ids.unsqueeze(0), loss_mask.unsqueeze(0), hidden_states, image_features

    # top-k 이미지 토큰 선택
    image_token_scores = last_token_attention[image_token_indices].float()
    topk = min(topk, image_token_scores.size(0))
    topk_indices_local = torch.topk(image_token_scores, topk).indices

    # 디바이스 일치시켜서 인덱싱
    image_token_indices = image_token_indices.to(topk_indices_local.device)
    topk_indices_global = image_token_indices[topk_indices_local]

    # 마스크 생성
    text_mask = input_ids != img_tok_index
    topk_img_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    topk_img_mask[topk_indices_global] = True
    final_mask = text_mask | topk_img_mask

    # 텐서 필터링
    filtered_input_ids = input_ids[final_mask].unsqueeze(0)
    filtered_loss_mask = loss_mask[final_mask].unsqueeze(0)
    filtered_hidden_states = hidden_states[final_mask]

    # 이미지 피처 필터링 (디바이스도 정렬)
    filtered_image_features = None
    if image_features is not None:
        if image_features.dim() == 3:
            image_features = image_features[0]
        image_features = image_features.to(device)
        filtered_image_features = image_features[topk_indices_local]

    return filtered_input_ids, filtered_loss_mask, filtered_hidden_states, filtered_image_features



def colorize_text(input_ids, loss_mask, tokenizer):
        # input_ids를 텍스트로 변환
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # 텍스트와 loss_mask를 이용하여 색상 적용
        colored_text = ""
        for token, mask in zip(tokens, loss_mask):
            if mask == 1:
                # loss_mask가 1인 토큰은 초록색
                colored_text += "\033[92m" + token + "\033[0m" + " "
            else:
                # loss_mask가 0인 토큰은 빨간색
                colored_text += "\033[91m" + token + "\033[0m" + " "

        print(colored_text)
        
def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    processor = AutoProcessor.from_pretrained('/home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369')
    image_folder = '/data/coco/train2017'
    #image_folder = '/data'
    
    #ds = load_dataset('json', data_files="/home/sangjun/EAGLE-LLAVA/playground/ShareGPT_V4.3_unfiltered_cleaned_split.json")
    ds = load_dataset('json', data_files="/home/sangjun/EAGLE-LLAVA/playground/llava_instruct_150k.json")
    #ds = load_dataset('json', data_files="/home/sangjun/dataset/sharegpt4v_instruct_gpt4-vision_cap100k.json")
    
    ds = ds['train']
    
    ds = ds.shuffle(seed=41)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    num_proc = 4
    
    
    def contains_special_token(turn, tokenizer, special_token_id=32000):
        input_ids = tokenizer(turn).input_ids

        return special_token_id in input_ids

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "image": [],
            "pixel_values":[],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            conv = get_conversation_template("vicuna")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            sorce= examples['conversations'][i]
            
            if roles[sorce[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                sorce = sorce[1:]
            conv.messages = []
            for j, sentence in enumerate(sorce):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()
            
            image_file = examples['image'][i]
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            inputs = processor(images=image, text=conversation, return_tensors="pt")
            input_ids=torch.as_tensor(inputs["input_ids"])[0]
            pixel_values=torch.as_tensor(inputs["pixel_values"])[0]
            loss_mask=torch.ones_like(input_ids)
            #print(i)

            sep = conv.sep + conv.roles[1] + ": "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())
            
            turns = conversation.split(conv.sep2)
            
            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                is_im_token = contains_special_token(turn,tokenizer)
                turn_len = len(tokenizer(turn).input_ids)
                if is_im_token : turn_len+=576

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                if is_im_token : instruction_len+=576
                
                if i==0:
                    instruction_len -= 1

                # image token length
                
                
                # Ignore the user instructions
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                if i==0:
                    cur_len -= 1
            loss_mask[cur_len:] = 0
            



            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["image"].append(image_file)
            new_examples["pixel_values"].append(pixel_values[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples
    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )
    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1

bigtokenizer = AutoProcessor.from_pretrained('/home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369').tokenizer
ds = build_dataset_rank(bigtokenizer)
print(ds)
bigmodel = LlavaForConditionalGeneration.from_pretrained(bigname,  device_map="cuda",torch_dtype=torch.float16, attn_implementation="eager")

bigmodel.eval()











@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    pixel_values=data["pixel_values"]
    loss_mask=data["loss_mask"]
    
    outs_big = bigmodel(input_ids.cuda(), pixel_values.cuda(), output_hidden_states=True, output_attentions=True)
    
    image_features = outs_big.image_hidden_states.cpu()
    hidden_state_big = outs_big.hidden_states[-1].cpu()
    
    
    input_ids, loss_mask, hidden_state_big, image_features = keep_topk_image_token(input_ids, loss_mask, hidden_state_big, image_features, outs_big.attentions)
    
    del outs_big
    
    # 캐시 정리 (가비지 컬렉션 + CUDA 캐시 해제)
    gc.collect()
    torch.cuda.empty_cache()
    td={"input_ids":input_ids.cpu()[0],"image":data["image"],"hidden_state":hidden_state_big.cpu(),"loss_mask":loss_mask.cpu()[0], "image_features":image_features.cpu()}
     # GPU 텐서는 여기서 더 안 쓸 거면 즉시 지워서 참조 해제
    #colorize_text(input_ids[0], loss_mask[0], bigtokenizer)
    del hidden_state_big
    gc.collect()
    torch.cuda.empty_cache()
    
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for data in tqdm(ds):
    # no_grad로 불필요한 그래디언트 계산 방지
    with torch.no_grad():
        outdata = ge(data)

    writedata(outdir, outdata)

    # outdata를 파일로 저장한 뒤 더는 메모리에 남길 필요가 없으면 지워줌
    del outdata
    gc.collect()
    torch.cuda.empty_cache()
