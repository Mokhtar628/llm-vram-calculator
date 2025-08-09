# 🧮 LLM VRAM Calculator

A simple Python command-line tool that estimates the **VRAM** required to serve a Large Language Model (LLM) using different settings like model size, quantization type, sequence length, and concurrent requests.  

This is particularly useful when working with frameworks like **vLLM**, **Text Generation Inference (TGI)**, or any custom LLM serving stack where you need to size your GPU memory requirements before deployment.

---

## 🚀 Why This Tool?
When deploying an LLM, one of the biggest questions is:
> **"Will my GPU have enough VRAM to serve this model?"**

This script gives you a quick, configurable estimate, so you can:
- Plan hardware requirements before purchase.
- Experiment with **different quantization levels** (`fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`).
- See the effect of **KV cache precision** on VRAM usage.
- Adjust for **different batch sizes** and **sequence lengths**.

---

## 📦 Features
- Calculates **model weights memory** based on parameters & quantization.
- Estimates **KV cache memory** based on layers, heads, sequence length, and concurrency.
- Accounts for **GPU overhead** (PyTorch, CUDA context, etc.).
- Fully configurable via **command-line arguments**.
- Provides a **clear breakdown** of each VRAM component.

---

## 📋 Installation
Clone the repository and make the script executable:

```bash
git clone .git
cd llm-vram-calculator
```

(Optional) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows
```

---

## 🛠 Usage

Run the script with your desired parameters:

```bash
python llm_vRAM_calculator.py     --model_params_billion 27     --model_quantization fp8     --max_seq_len 2048     --max_concurrent_requests 32     --num_layers 48     --kv_attention_heads 8     --head_dim 128     --kv_cache_quantization fp8     --overhead_gb 3.0
```

---

## 📊 Example Output

```
--- LLM VRAM Requirement Estimation ---
Model Parameters: 27.0 Billion
Model Quantization: FP8
KV Cache Quantization: FP8
Max Sequence Length: 2048
Max Concurrent Requests: 32
Number of Layers: 48
KV Attention Heads: 8
Head Dimension: 128
Max Total Tokens in KV Cache: 65536

1. Model Weights Memory: 25.15 GB
2. KV Cache Memory:     6.0 GB
3. Other Overheads:     3.0 GB (estimated)
----------------------------------------
Total Estimated VRAM:   34.15 GB
----------------------------------------

Note: This is an estimation. Actual VRAM usage may vary.
Consider adding a buffer for robust serving.
```

---

## ⚙️ Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model_params_billion` | float | ✅ | Number of model parameters in billions. Example: `27` for 27B. |
| `--model_quantization` | str | ✅ | Quantization precision: `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`. |
| `--max_seq_len` | int | ✅ | Max sequence length (prompt + generation). |
| `--max_concurrent_requests` | int | ✅ | Max number of concurrent requests. |
| `--num_layers` | int | ❌ | Transformer layers. Default: `48` for 27B models. |
| `--kv_attention_heads` | int | ❌ | Number of KV attention heads (GQA). Default: `8`. |
| `--head_dim` | int | ❌ | Dimensionality of each attention head. Default: `128`. |
| `--kv_cache_quantization` | str | ❌ | Quantization for KV cache. Defaults to model quantization. |
| `--overhead_gb` | float | ❌ | Estimated GPU overhead in GB. Default: `3.0`. |

---

## 🧠 How It Works
1. **Model Weights Memory** – Calculated from:
   ```
   (Parameters × Bytes per Parameter) / (1024³)
   ```
2. **KV Cache Memory** – Based on:
   ```
   2 × Layers × KV Heads × Head Dim × Total Tokens × Bytes per Value
   ```
3. **Overhead** – Fixed buffer for CUDA, PyTorch, and other ops.
4. **Total VRAM** = Weights + KV Cache + Overhead.

---

## 💡 Tips
- If your GPU memory is close to the estimated VRAM, reduce:
  - **Sequence length** (`--max_seq_len`)
  - **Concurrent requests** (`--max_concurrent_requests`)
  - Or use **lower quantization** (`int8` or `int4`).
- Always leave **2–3 GB of VRAM buffer** for stability.


**Happy LLM Deploying 🚀**
