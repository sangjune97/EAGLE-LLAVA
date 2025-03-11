import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir1')
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

bigname="llava-hf/llava-1.5-13b-hf"
#bigname="lmsys/vicuna-13b-v1.5"

def get_image_features(bigmodel, pixel_values: torch.FloatTensor):
            """
            Obtains image last hidden states from the vision tower and apply multimodal projection.

            Args:
                pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
                   The tensors corresponding to the input images.
                vision_feature_layer (`int`):
                    The index of the layer to select the vision feature.
                vision_feature_select_strategy (`str`):
                    The feature selection strategy used to select the vision feature from the vision backbone.
                    Can be one of `"default"` or `"full"`
            Returns:
                image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
            """
            
            vision_feature_layer = bigmodel.config.vision_feature_layer
            vision_feature_select_strategy = bigmodel.config.vision_feature_select_strategy
            image_outputs = bigmodel.vision_tower(pixel_values, output_hidden_states=True)
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(f"Unexpected select feature strategy: {bigmodel.config.vision_feature_select_strategy}")
            image_features = bigmodel.multi_modal_projector(selected_image_feature)
            return image_features
        
def remove_image_token(input_ids, img_tok_index, loss_mask, hidden_states=None):
    mask = input_ids != img_tok_index
    filtered_input_ids = input_ids[mask].view(1, -1).to(input_ids.device)
    filtered_loss_mask = loss_mask[mask].view(1, -1).to(input_ids.device)
    if hidden_states is not None:
        filtered_hidden_states = hidden_states[:, mask[0], :]
        return filtered_input_ids, filtered_loss_mask, filtered_hidden_states
    
    return filtered_input_ids, filtered_loss_mask


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
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-13b-hf')
    
    ds = load_dataset('json', data_files="/home/sangjun/EAGLE-LLAVA/playground/ShareGPT_V4.3_unfiltered_cleaned_split.json")
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

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            conv = get_conversation_template("vicuna")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=4096,
                truncation=True,
            ).input_ids[0]
            
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
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            loss_mask[cur_len:] = 0
            #colorize_text(input_ids,loss_mask, tokenizer)
            



            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
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

#bigtokenizer = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf').tokenizer
bigtokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-13b-hf",use_fast=False)
ds = build_dataset_rank(bigtokenizer)
print(ds)
bigmodel = LlavaForConditionalGeneration.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
bigmodel.eval()











@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
 
    td={"input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}
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


