cifar100_classes = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]
imagenet_templates2 = [
    'itap of a {}.',
    'a bad photo of the {}.',
    'a origami {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
]

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
from eagle.model.ea_model import EaModel
from transformers import AutoProcessor, CLIPModel, CLIPProcessor

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
clipname = "openai/clip-vit-large-patch14-336"
clipmodel = CLIPModel.from_pretrained(clipname,device_map="auto")
clipprocessor = CLIPProcessor.from_pretrained(clipname)
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def clip(images, coco_classes, imagenet_templates, clipmodel, clipprocessor):
        clipmodel.eval()
        with torch.no_grad():
            device = next(clipmodel.parameters()).device

            all_texts = []
            for class_name in coco_classes:
                texts = [template.format(class_name) for template in imagenet_templates]
                all_texts.extend(texts)

            # 텍스트 토큰화 (한번만 하면 돼)
            text_inputs = clipprocessor(text=all_texts, padding=True, return_tensors="pt").to(device)
            text_features = clipmodel.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

            image_inputs = clipprocessor(images=images, return_tensors="pt").to(device)
            image_features = clipmodel.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

            # 유사도 측정 [batch, 텍스트 개수]
            logits = image_features @ text_features.T

            # 가장 유사한 텍스트 찾기
            max_indices = logits.argmax(dim=-1).tolist()

            best_templates = [all_texts[idx] for idx in max_indices]

        return best_templates
    
def insert_clip_output_after_image(conversation: str, clip_output: str) -> str:
        marker = "<image>"
        if marker in conversation:
            # 첫 번째 <image> 뒤에 clip_output 추가
            return conversation.replace(marker, marker + " " + clip_output, 1)
        else:
            # <image>가 없으면 원본 문자열 그대로 반환
            return conversation
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
    #yuhuili/EAGLE-Vicuna-7B-v1.3
    model.eval()
    clipmodel.eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        clip_outputs = clip(image, cifar100_classes, imagenet_templates2, clipmodel, clipprocessor)
        prompt = insert_clip_output_after_image(prompt, clip_outputs[0])
        
        inputs = processor(images=image, text=prompt, return_tensors='pt')
        
        # **시간 측정 시작**
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.inference_mode():
            output_ids, _ , _ , avg_accept_length = model.eagenerate(
                input_ids=torch.as_tensor(inputs["input_ids"]).cuda(), 
                pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda(),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=1024,
                log=True,
                token_process=args.token_process)
            
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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--token-process", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
