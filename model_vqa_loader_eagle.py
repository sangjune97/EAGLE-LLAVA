import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
from eagle.model.ea_model import EaModel
from transformers import AutoProcessor

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
def start_timer():
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end

def end_timer(start, end, name="블록"):
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    return elapsed


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.model_config = model_config
        self.processor = processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        return inputs

    def __len__(self):
        return len(self.questions)
    
def collate_fn(batch):
    inputs = batch[0]  # 첫 번째 요소 가져오기 (batch_size=1일 경우)
    inputs = {k: v.squeeze(0).unsqueeze(0) if len(v.shape) == 3 else v for k, v in inputs.items()}
    return inputs

# DataLoader
def create_data_loader(questions, image_folder, processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    ea_model_path = os.path.expanduser(args.ea_model_path)
    model_name = get_model_name_from_path(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    #tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
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
    # **시간 측정 시작**
    start, end = start_timer()#timer start
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, processor, model.config)
    # **시간 측정 종료**
    encoding_time = end_timer(start, end, name="total")#timer end

    for (inputs), line in tqdm(zip(data_loader, questions), total=len(questions)):
        # **시간 측정 시작**
        start, end = start_timer()#timer start
        idx = line["question_id"]
        cur_prompt = line["text"]

        

        with torch.inference_mode():
            output_ids, _, _, avg_accept_length, initialize_time, initialize_tree_time, tree_decode_total_time, evaluate_posterior_total_time, update_inference_inputs_total_time,_,_ = model.eagenerate(
                input_ids=torch.as_tensor(inputs["input_ids"]).cuda(), 
                pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda(),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=1024,
                log=True,
                token_process=args.token_process,
                num_img_tokens = args.num_img_tok,
                )
            
        # **시간 측정 종료**
        decoding_time = end_timer(start, end, name="total")#timer end
        
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        num_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]

        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "encoding_time": encoding_time,
                                   "decoding_time": decoding_time,
                                   "num_tokens": num_tokens,
                                   "avg_accept_length":avg_accept_length,
                                   "initialize_time":initialize_time, 
                                   "initialize_tree_time":initialize_tree_time, 
                                   "tree_decode_total_time":tree_decode_total_time, 
                                   "evaluate_posterior_total_time":evaluate_posterior_total_time, 
                                   "update_inference_inputs_total_time":update_inference_inputs_total_time,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
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
    parser.add_argument("--num_img_tok", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
