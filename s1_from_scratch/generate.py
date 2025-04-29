from vllm import LLM, SamplingParams # compared to transformers, vllm is easier to add stop
from transformers import AutoTokenizer

MAX_TOKENS = 4096
# load original model DS-R1-Distill-Qwen1.5B
tokenizer = AutoTokenizer.from_pretrained("./DeepSeek-R1-Distill-Qwen-1.5B")
llm = LLM(model="./DeepSeek-R1-Distill-Qwen-1.5B", gpu_memory_utilization=0.95)

# load LoRA model Qwen0.5B
# from peft import PeftModel
# tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct")
# llm = LLM(model="./Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.95)
# llm = PeftModel.from_pretrained(llm, "./s1") # load lora 

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=MAX_TOKENS,
    skip_special_tokens=False
)

# prompt = '9.11和9.8谁大？'
# prompt = 'which is bigger, 9.11 or 9.8?'
prompt = 'how many r in strawberry?'
# Qwen template
prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"

# 模型原始输出部分
outputs = llm.generate(
    prompt,
    sampling_params
)
print(f'原始输出：{prompt}{outputs[0].outputs[0].text}')
print('+'*20)

# budget forcing 部分
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=MAX_TOKENS,
    stop='</think>', # 在输出中添加stop token
    skip_special_tokens=False
)

outputs = llm.generate(
        prompt,
        sampling_params
    )
wait = 'Wait'
for i in range(1): # #times to skip stop token
    prompt += outputs[0].outputs[0].text + wait

    outputs = llm.generate(
        prompt,
        sampling_params
    )

print(f'wait后的输出：{prompt}{outputs[0].outputs[0].text}')
print('+'*20)
prompt += outputs[0].outputs[0].text
stop_token_ids = tokenizer("<|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)
outputs = llm.generate(
    prompt,
    sampling_params=sampling_params,
)

print(f'最后的输出：{prompt}{outputs[0].outputs[0].text}')



