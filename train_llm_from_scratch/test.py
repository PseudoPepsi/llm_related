from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from train import LLM, Config


t = AutoTokenizer.from_pretrained('./saves/model')
AutoConfig.register("small_model", Config)
AutoModelForCausalLM.register(Config, LLM)

model = AutoModelForCausalLM.from_pretrained('saves/dpo-1-epoch')

input_data = t.apply_chat_template([{'role':'user', 'content':'1+1等于几'}])
print(input_data)

for token in model.generate({"input_ids":torch.tensor(input_data).unsqueeze(0), "labels":None}, t.eos_token_id, 200, stream=False,temperature=0.0, top_k=8):
    print(t.decode(token[0]))