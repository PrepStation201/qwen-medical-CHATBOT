import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr

# Load fine-tuned model and tokenizer
adapter_path = "./qwen_medical_finetuned"
base_model = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

def chat(instruction):
    prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("assistant\n")[-1].strip()

gr.Interface(fn=chat,
             inputs=gr.Textbox(label="Ask something medical"),
             outputs=gr.Textbox(label="Response"),
             title="🧠 Qwen Medical Chatbot",
             description="LoRA fine-tuned Qwen2.5-0.5B model on medical Q&A").launch()
