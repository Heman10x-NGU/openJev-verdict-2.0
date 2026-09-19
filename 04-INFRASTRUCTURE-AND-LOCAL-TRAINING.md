# Infrastructure, compute options, and budget

This document outlines the compute requirements, memory footprints, and practical deployment options across Apple Silicon, free cloud notebooks, and domestic cloud GPUs in India.

## Hardware requirements and memory math

Training a 139M parameter encoder (such as ModernBERT-base) requires modest compute compared to multi-billion parameter autoregressive models.

### Memory footprint during training
* Model parameters (FP16): 278 MB
* Model gradients (FP16): 278 MB
* AdamW optimizer states (FP32 first and second moments): 1,112 MB
* Forward activations (batch size 32, sequence length 512): ~800 MB
* **Total training memory footprint:** ~2.5 GB of VRAM

This fits inside any modern GPU or laptop unified memory architecture without memory paging.

---

## Option 1: Local training on MacBook Air M4 (16GB RAM)

The M4 chip features unified memory with 120 GB/s bandwidth.

### Execution environment
Install Apple MLX and Hugging Face dependencies:
```bash
pip install mlx mlx-lm transformers datasets
```

### Advantages
* Zero cloud expense.
* Fast unified memory transfers without PCIe bus bottlenecks.
* Direct integration with macOS testing harnesses.

### Thermal considerations
The MacBook Air is fanless. Continuous multi-core GPU execution heats the aluminum chassis, triggering thermal throttling that reduces clock frequencies by 15% to 20% after 20 minutes of sustained load. For training runs longer than 45 minutes, elevate the laptop on an aluminum stand or point an external desk fan at the underside.

---

## Option 2: Free cloud notebooks (Kaggle)

Kaggle provides 30 hours per week of free GPU runtime.

### Specifications
* GPU: Nvidia Tesla T4 (16GB GDDR6) or Nvidia P100 (16GB HBM2).
* Framework: PyTorch with native CUDA.

### Workflow
1. Upload your synthetic dataset as a private Kaggle Dataset.
2. Launch a Jupyter notebook with the GPU T4 accelerator enabled.
3. Train with `torch.cuda.amp.autocast()` enabled for mixed-precision acceleration.
4. Export the resulting checkpoint to Hugging Face Hub.

---

## Option 3: Rented cloud GPUs in India

If you require dedicated high-throughput compute with domestic payment support, several platforms operate directly in India:

### 1. JarvisLabs.ai
* Headquarters and data centers: India and Europe.
* Payment methods: Domestic Indian credit cards, debit cards, net banking, and UPI.
* Pricing (on-demand per-minute billing):
  * Nvidia A100 (40GB VRAM): $0.89 per hour (roughly ₹75 per hour)
  * Nvidia A100 (80GB VRAM): $1.49 per hour (roughly ₹140 per hour)
  * Nvidia RTX A5000: $0.59 to $1.29 per hour

### 2. E2E Networks
* Headquarters: India.
* Payment methods: INR invoicing, domestic banking, and Indian payment gateways.
* Offerings: Nvidia L40S and A100 instances for enterprise workloads.

---

## Complete project budget

Training an initial production-grade checkpoint on 100,000 synthetic samples requires minimal capital:

| Expense item | Provider | Quantity | Estimated cost |
| :--- | :--- | :--- | :--- |
| Synthetic dataset generation (50k pairs) | OpenAI (GPT-4o-mini API) | 15M input tokens | $2.25 |
| Model training (ModernBERT, 100k steps) | JarvisLabs (Nvidia A100 40GB) | 1.5 hours | $1.35 (₹115) |
| Model weights hosting | Hugging Face Hub | Unlimited public storage | Free |
| WebGPU static demo hosting | GitHub Pages | Static web assets | Free |
| **Total project cost** | | | **$3.60 (roughly ₹300)** |
