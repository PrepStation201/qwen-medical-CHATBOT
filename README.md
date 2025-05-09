# qwen-medical-CHATBOT
```markdown
# 🧠 Qwen Medical Chatbot

This project presents a domain-specific chatbot fine-tuned on healthcare and alternative medicine instructions using Qwen2.5-0.5B-Instruct and LoRA (Low-Rank Adaptation). It leverages efficient fine-tuning and an interactive Gradio interface.

## 🚀 Features
- Fine-tuned LLM specialized for medical/alternative health Q&A
- Efficient LoRA-based training (low memory requirement)
- Gradio UI for live interaction
- Quantized 4-bit model to support resource-constrained environments

## 🧰 Technologies Used
- 🤗 Hugging Face Transformers & TRL (Trainer)
- 🧪 LoRA (via PEFT)
- 📊 Matplotlib + TensorBoard (training visualization)
- 🌐 Gradio (web interface)

## 📁 Dataset
Custom dataset of instruction-response pairs (`nlpsquad_dataset.csv`) containing natural queries related to alternative treatments, Ayurveda, herbal use, and nutritional advice.

## 🧠 Model Architecture
![architecture](architecture.png)

## 🧪 Before vs After Fine-Tuning
See [comparison_outputs.md](comparison_outputs.md) for sample output comparisons between the base model and fine-tuned version.

## 📊 Visualizations
- Training Loss over Time (`loss_curve.png`)
- Learning Rate Schedule (`lr_curve.png`)
- Input Instruction Length Distribution (`instr_length_hist.png`)

## 🛠️ Running the Project Locally
```bash
pip install -r requirements.txt
python app.py
```
