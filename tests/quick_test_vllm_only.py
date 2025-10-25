import torch
from vllm import LLM, SamplingParams

# Use only GPU 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Model and settings
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    dtype="float16",
    max_seq_len_to_capture=1024,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)

prompt = "What is the capital of France?"
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

outputs = llm.generate([prompt], sampling_params)
for output in outputs:
    print(f"Prompt: {prompt}\nOutput: {output.outputs[0].text}") 