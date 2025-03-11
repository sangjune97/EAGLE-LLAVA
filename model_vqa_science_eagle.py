import argparse
import torch
import os
import json
import time
from tqdm import tqdm
import shortuuid
from eagle.model.ea_model import EaModel
from transformers import AutoProcessor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    ea_model_path = os.path.expanduser(args.ea_model_path)
    model_name = get_model_name_from_path(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = EaModel.from_pretrained(
            base_model_path=model_path,
            ea_model_path=ea_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=60,
            depth=5
    )
    model.eval()
    
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            image = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = processor(images=image, text=prompt, return_tensors='pt')
        # **시간 측정 시작**
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
                if image is not None:
                    output_ids, _ , _ , avg_accept_length = model.eagenerate(
                        input_ids=torch.as_tensor(inputs["input_ids"]).cuda(), 
                        pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda(),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=1024,
                        log=True,
                        token_process=args.token_process
                        )
                else :
                    output_ids, _ , _ , avg_accept_length = model.eagenerate(
                        input_ids=torch.as_tensor(inputs["input_ids"]).cuda(), 
                        pixel_values=None,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=1024,
                        log=True,
                        token_process=args.token_process
                        )
        # **시간 측정 종료**
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        num_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
        tok_per_sec = num_tokens/total_time

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "total_time": total_time,
                                   "num_tokens": num_tokens,
                                   "tok_per_sec": tok_per_sec,   
                                   "avg_accept_length":avg_accept_length.item(),
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--ea-model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--token-process", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
