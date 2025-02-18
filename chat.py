import os

new_dir = os.getcwd()
model_dir = f'{new_dir}/model/Qwen/Qwen-1_8B-Chat'

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

print("正在从本地加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参
print("完成本地模型的加载")

prompt = "你好"

response = model.chat(tokenizer, prompt, history=[])
print(response)