"""test"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM
import torch
from torch.nn import functional as F

# load model
device = "cuda:1"
processor = AutoProcessor.from_pretrained("./siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained('./Qwen2.5-0.5B-Instruct')
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)
# model = AutoModelForCausalLM.from_pretrained('./save/pretrain')
model = AutoModelForCausalLM.from_pretrained('./save/sft')
model.to(device)

# prepare input
# q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":'图片里的人是男是女\n<image>'}], \
# q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":'图片里有几个人n<image>'}], \
q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":'描述图片内容\n<image>'}], \
            tokenize=False, \
            add_generation_prompt=True).replace('<image>', '<|image_pad|>'*49) # 49 = (224/16)^2/4
input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
input_ids = input_ids.to(device)
image = Image.open('./小狗美女海边-Dog-Woman-Sea.jpg').convert("RGB")
pixel_values = processor(text=None, images=image).pixel_values # (1, 3, process_size=224, 224)
pixel_values = pixel_values.to(device)
model.eval()

# inference params
max_new_tokens = 100
# temperature = 0.0
temperature = 0.7
eos = tokenizer.eos_token_id
top_k = None

# model inference
s = input_ids.shape[1] # length of input tokens
while input_ids.shape[1] < s + max_new_tokens - 1:  
    inference_res = model(input_ids, None, pixel_values)  
    logits = inference_res.logits 
    logits = logits[:, -1, :]  # take last output token

    for token in set(input_ids.tolist()[0]):  
        logits[:, token] /= 1.0

    if temperature == 0.0: 
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        logits = logits / temperature  
        if top_k is not None:  
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf') 

        probs = F.softmax(logits, dim=-1)  
        idx_next = torch.multinomial(probs, num_samples=1, generator=None) # sample next token

    if idx_next == eos:  
        break

    input_ids = torch.cat((input_ids, idx_next), dim=1) # concat next token to input_id
print(tokenizer.decode(input_ids[:, s:][0]))