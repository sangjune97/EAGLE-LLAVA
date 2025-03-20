import argparse
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

bigname="llava-hf/llava-1.5-7b-hf"
#bigname="lmsys/vicuna-13b-v1.5"
        
def remove_image_token(input_ids, img_tok_index, loss_mask, hidden_states=None):
    mask = input_ids != img_tok_index
    filtered_input_ids = input_ids[mask].view(1, -1).to(input_ids.device)
    filtered_loss_mask = loss_mask[mask].view(1, -1).to(input_ids.device)
    if hidden_states is not None:
        filtered_hidden_states = hidden_states[:, mask[0], :]
        return filtered_input_ids, filtered_loss_mask, filtered_hidden_states
    
    return filtered_input_ids, filtered_loss_mask

def pool_24x24_to_12x12(
    img_block_states: torch.Tensor,
    pool_type: str = "max"
) -> torch.Tensor:
    """
    img_block_states: shape [576, 4096] (24*24=576, hidden_dim=4096)
    -> (24,24,4096)로 보고 2D 풀링(kernel=2) -> (12,12,4096) -> 최종 (144,4096).

    pool_type: "max" 또는 "mean"
      - "max"  -> F.max_pool2d
      - "mean" -> F.avg_pool2d
    """
    # (576, 4096) -> (24,24,4096)
    reshaped_3d = img_block_states.view(24, 24, 4096)
    
    # 2D 풀링을 위해 NCHW 형식으로 바꾸기: [N=1, C=4096, H=24, W=24]
    reshaped_4d = reshaped_3d.permute(2, 0, 1).unsqueeze(0)
    # shape: [1, 4096, 24, 24]

    # 2D 풀링 (kernel_size=2)
    if pool_type == "max":
        pooled_4d = F.max_pool2d(reshaped_4d, kernel_size=2)
    elif pool_type == "mean":
        pooled_4d = F.avg_pool2d(reshaped_4d, kernel_size=2)
    else:
        raise ValueError(f"Unsupported pool_type: {pool_type}. Use 'max' or 'mean'.")

    # pooled_4d: [1, 4096, 12, 12]

    # 다시 (12,12,4096)로 (N=1 차원 제거, permute)
    pooled_3d = pooled_4d.squeeze(0).permute(1, 2, 0)
    # shape: (12,12,4096)

    # Flatten -> (144, 4096)
    flattened_2d = pooled_3d.view(-1, 4096)
    return flattened_2d

def pool_tensor(input_tensor):
    # 입력 텐서의 형태를 [1, 24, 24, 4096]으로 변경
    reshaped_tensor = input_tensor.reshape(1, 24, 24, 4096)

    # 차원을 (배치 크기, 채널, 높이, 너비)로 변환
    corrected_tensor = reshaped_tensor.permute(0, 3, 1, 2)  # 이제 (1, 4096, 24, 24) 형태

    # 최대 풀링 적용
    pooled_tensor = F.max_pool2d(corrected_tensor, kernel_size=2, stride=2, padding=0)

    # pooled_tensor를 [1, 144, 4096]으로 다시 변환
    reshaped_back_tensor = pooled_tensor.reshape(1, 144, 4096)

    return reshaped_back_tensor

def pool_image_token(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    img_tok_index: int = 3200,
    img_token_loss: int = 0,
    pool_type: str = "max"
):
    """
    원본 시퀀스( batch=1 )에서
    - 연속된 이미지 토큰(img_tok_index) 구간을 추출,
    - (block_size, hidden_dim) -> 2D 풀링 -> (144, hidden_dim) 변환,
    - 변환된 144개를 시퀀스에 그대로 삽입.

    Args:
        input_ids: (1, seq_len)
        loss_mask: (1, seq_len)
        hidden_states: (1, seq_len, hidden_dim)
        img_tok_index: 이미지 토큰 ID (기본=3200)
        img_token_loss: 풀링 후 삽입되는 새 토큰들의 loss_mask 값(1 또는 0)
        pool_type: "max" or "mean" (2D 풀링 방식)

    Returns:
        new_input_ids: (1, new_seq_len)
        new_loss_mask: (1, new_seq_len)
        new_hidden_states: (1, new_seq_len, hidden_dim)
    """
    device = input_ids.device

    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "batch_size=1 가정"
    assert hidden_states.dim() == 3 and hidden_states.size(0) == 1, "batch_size=1 가정"

    orig_ids = input_ids[0]       # shape: (seq_len,)
    orig_loss = loss_mask[0]      # shape: (seq_len,)
    orig_hid = hidden_states[0]   # shape: (seq_len, hidden_dim)

    new_ids_list = []
    new_loss_list = []
    new_hid_list = []

    seq_len = orig_ids.size(0)
    i = 0
    while i < seq_len:
        if orig_ids[i] == img_tok_index:
            # 연속된 이미지 토큰 구간 찾기
            start = i
            while i < seq_len and orig_ids[i] == img_tok_index:
                i += 1
            end = i  # [start, end) 구간 전부 img_tok_index

            # 이미지 토큰 구간 히든 스테이트: shape=(block_size, hidden_dim)
            img_block_states = orig_hid[start:end]

            # 2D 풀링 -> (144, hidden_dim)
            pooled_2d = pool_24x24_to_12x12(img_block_states, pool_type=pool_type)
            # shape: [144, 4096] (가정)

            # 새 시퀀스에 144개 토큰 삽입
            # - 각 행 = 1개의 토큰
            # - 토큰ID = img_tok_index
            # - loss_mask = img_token_loss
            num_tokens = pooled_2d.size(0)  # 144
            for row_idx in range(num_tokens):
                new_ids_list.append(torch.tensor([img_tok_index], device=device))
                new_loss_list.append(torch.tensor([img_token_loss], device=device))
                row_vec = pooled_2d[row_idx].unsqueeze(0)  # (1, hidden_dim)
                new_hid_list.append(row_vec)

        else:
            # 이미지 토큰이 아니면 그대로
            new_ids_list.append(orig_ids[i].unsqueeze(0))   
            new_loss_list.append(orig_loss[i].unsqueeze(0))
            new_hid_list.append(orig_hid[i].unsqueeze(0))
            i += 1

    # 리스트 -> 최종 텐서
    new_input_ids = torch.cat(new_ids_list, dim=0).unsqueeze(0)      # (1, new_seq_len)
    new_loss_mask = torch.cat(new_loss_list, dim=0).unsqueeze(0)     # (1, new_seq_len)
    new_hidden_states = torch.cat(new_hid_list, dim=0).unsqueeze(0)  # (1, new_seq_len, hidden_dim)

    return new_input_ids, new_loss_mask, new_hidden_states

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
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    image_folder = '/data/COCO/train2017'
    
    #ds = load_dataset('json', data_files="/home/sangjun/EAGLE-LLAVA/playground/ShareGPT_V4.3_unfiltered_cleaned_split.json")
    ds = load_dataset('json', data_files="/home/sangjun/EAGLE-LLAVA/playground/llava_instruct_150k.json")
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    num_proc = 4
    
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
    
    def contains_special_token(turn, tokenizer, special_token_id=32000):
        input_ids = tokenizer(turn).input_ids

        return special_token_id in input_ids
    
    def remove_elements(tensor, value, num_elements):
        # 값이 처음 발견되는 인덱스를 찾습니다.
        index = (tensor == value).nonzero(as_tuple=True)[0]
        if index.nelement() == 0:
            # 값이 텐서에 없다면 원본 텐서를 반환합니다.
            return tensor
        # 값이 있는 인덱스를 기준으로 432개의 엘리먼트를 삭제합니다.
        start_index = index[0].item()
        end_index = start_index + num_elements
        if end_index > tensor.nelement():
            # 만약 end_index가 텐서의 크기를 초과한다면 start_index부터 끝까지 삭제합니다.
            return torch.cat((tensor[:start_index], torch.tensor([])), dim=0)
        else:
            # 그렇지 않다면 start_index부터 end_index 이전까지 삭제합니다.
            return torch.cat((tensor[:start_index], tensor[end_index:]), dim=0)

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

bigtokenizer = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf').tokenizer
ds = build_dataset_rank(bigtokenizer)
print(ds)
bigmodel = LlavaForConditionalGeneration.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)

bigmodel.eval()



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
        
@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    pixel_values=data["pixel_values"]
    loss_mask=data["loss_mask"]
    outs_big = bigmodel(input_ids.cuda(), pixel_values.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    filtered_input_ids, filtered_loss_mask, filtered_hidden_states = pool_image_token(input_ids, loss_mask, hidden_state_big, 32000,0,"mean")
    image_features = pool_tensor(outs_big.image_hidden_states)
    
    td={"input_ids":filtered_input_ids.cpu()[0],"image":data["image"],"hidden_state":filtered_hidden_states.cpu()[0],"loss_mask":filtered_loss_mask.cpu()[0],"image_features":image_features.cpu()[0]}
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
    outdata = ge(data)
    writedata(outdir,outdata)


