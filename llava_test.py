from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
import os
import torch
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
qs = "What type of elephant is being transported on the truck?"#url = "https://media.istockphoto.com/id/92450205/photo/sunsplashed-window.jpg?s=612x612&w=0&k=20&c=dTuhETbiWnoxAR1Ek5ROlj0liKxBazb14d9rsfe4XTc="
#image = Image.open(requests.get(url, stream=True).raw)
#image = Image.open(os.path.join("/home/sangjun/LLaVA/playground/data/eval/mm-vet/images/v1_30.jpg")).convert('RGB')
image = Image.open(os.path.join("/data/COCO/train2017/000000027989.jpg")).convert('RGB')

cur_prompt = qs
qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
conv = conv_templates['llava_v1'].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

image = Image.open(os.path.join("/data/COCO/train2017/000000027989.jpg")).convert('RGB')
image_tensor = process_images([image], image_processor, model.config)[0]
print(prompt)
print(input_ids)
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(),
        image_sizes=[image.size],
        # no_repeat_ngram_size=3,
        max_new_tokens=1024,
        use_cache=True)

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)