# Llama3.1-8B Fine-tuning with QLoRA

## About

This project demonstrates how to fine-tune Meta's Llama3.1-8B model using QLoRA (Quantized Low-Rank Adaptation) on your own custom datasets. QLoRA is a parameter-efficient fine-tuning technique that allows you to fine-tune large language models using 4-bit quantization and low-rank adapters, significantly reducing memory requirements while maintaining performance.

### Why Fine-tune on Personal Datasets?

Fine-tuning LLMs on your personal or domain-specific data offers several compelling advantages:

- **Domain Expertise**: Teach the model specialized knowledge from your field (medical, legal, technical, etc.)
- **Custom Writing Style**: Adapt the model to match your organization's tone and communication style
- **Proprietary Knowledge**: Incorporate private data that wasn't part of the original training
- **Improved Accuracy**: Get better results on tasks specific to your use case
- **Data Privacy**: Keep sensitive information within your control during training

### Fine-tuning vs RAG: When to Choose What?

| Aspect | Fine-tuning | RAG (Retrieval-Augmented Generation) |
|--------|-------------|--------------------------------------|
| **Knowledge Integration** | Permanently learns from data | Retrieves relevant info at query time |
| **Model Size** | Same model size, new weights | Base model + external knowledge base |
| **Update Frequency** | Requires retraining for updates | Real-time knowledge updates |
| **Computational Cost** | High training cost, low inference | Low training cost, higher inference |
| **Data Privacy** | Data integrated into model weights | External database queries |
| **Use Case** | Domain expertise, style adaptation | Dynamic facts, large knowledge bases |

**Choose Fine-tuning when:**
- You need domain-specific expertise baked into the model
- You want consistent writing style/tone
- You have limited, focused datasets
- You need offline operation without external databases

**Choose RAG when:**
- You need real-time, updateable information
- You have large, dynamic knowledge bases
- You want to maintain data separation from the model
- You need to cite specific sources

### Example Dataset: Hawaiian Wildfire Data

In this tutorial, we use the **Hawaiian Wildfire dataset from [Polo Club of Data Science, Georgia Tech](https://github.com/poloclub)** as a practical example. This dataset contains detailed reports about the 2023 Hawaiian wildfires, demonstrating how to fine-tune a model on recent events that occurred after the base model's training cutoff. This serves as a perfect example of incorporating new, domain-specific knowledge into an existing LLM.

The Polo Club of Data Science specializes in Human-Centered AI, AI Security, Visual Analytics, and Graph Mining & Visualization, making their datasets excellent examples for educational fine-tuning projects.

### Key Features
- **Memory Efficient**: Uses 4-bit quantization to reduce memory footprint
- **Parameter Efficient**: QLoRA adapters require minimal additional parameters
- **High Performance**: Maintains model quality while being computationally efficient
- **Cloud GPU Optimized**: Designed for Google Colab A100 GPU environment

### What You'll Learn
- How to set up a QLoRA fine-tuning pipeline
- Memory optimization techniques for large language models
- Parameter-efficient fine-tuning strategies
- Best practices for working with Llama3-8B

## Prerequisites

- Python 3.8 or higher
- Google Colab with A100 GPU (40GB VRAM) - recommended for optimal performance
- Git
- [UV package manager](https://github.com/astral-sh/uv) (recommended for fast dependency management)

## Installation Guide

### Option 1: Google Colab with A100 GPU (Recommended)

For the best experience, use Google Colab with A100 GPU for optimal performance and CUDA memory:

#### 1. Clone the Repository in Colab
```bash
!git clone https://github.com/your-username/Llama3-8b-Finetuning-QLoRA.git
%cd Llama3-8b-Finetuning-QLoRA
```

#### 2. Install Dependencies with Pip
```bash
!pip install peft accelerate bitsandbytes datasets evaluate transformers torch
```

*Note: Virtual environments are not needed in Colab as each session is isolated.*

### Option 2: Local Development with GPU

For local development with your own GPU hardware:

#### 1. Install UV (Optional but Recommended)
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

#### 2. Clone the Repository
```bash
git clone https://github.com/StevenD24/Llama3-8b-Finetuning-QLoRA.git
cd Llama3-8b-Finetuning-QLoRA
```

#### 3. Initialize Virtual Environment
```bash
# Create a virtual environment
uv venv  # or: python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### 4. Install Dependencies
```bash
# With UV (faster)
uv pip install peft accelerate bitsandbytes datasets evaluate transformers torch jupyter ipykernel

# Or with standard pip
pip install peft accelerate bitsandbytes datasets evaluate transformers torch jupyter ipykernel
```

*Note: Requires CUDA-compatible GPU with 16GB+ VRAM for optimal performance.*

## Quick Start

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**:
   Open `finetuning_llama3_8b_qlora.ipynb` in your browser

3. **Follow the notebook cells**:
   - The notebook guides you through the entire fine-tuning process
   - Execute cells in order to set up the model, data, and training pipeline

## Data Flow During Training

Understanding how data flows through the training pipeline is crucial for debugging and optimization:

```
Data Processing Pipeline:
Raw Text Files → Tokenization → Collation → Batching
                                              ↓
Training Loop:
Forward Pass → Calculate Loss → Accumulate Gradients (4x) → Update Weights
     ↑                                                            ↓
Save Checkpoints ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
(every epoch)
```

### Data Processing Steps:
1. **Raw Text**: Hawaiian wildfire data files are loaded as plain text
2. **Tokenization**: Text is converted to token IDs using Llama tokenizer
3. **Collation**: Variable-length sequences are padded to uniform length
4. **Batching**: Data is grouped into batches of 2 samples per GPU
5. **Forward Pass**: Model processes batches to generate predictions
6. **Loss Calculation**: Compare predictions with target tokens
7. **Gradient Accumulation**: Gradients are accumulated over 4 steps (effective batch size = 8)
8. **Weight Updates**: Model parameters are updated using 8-bit AdamW optimizer
9. **Checkpointing**: Model state is saved after each epoch for recovery

## Usage Tips

### Package Management
- **For Local Development**: Use UV for faster dependency resolution and installation
- **For Google Colab**: Use standard pip commands (`!pip install package_name`) for compatibility
- **For sharing notebooks**: Always use pip commands for maximum compatibility across environments

### Memory Optimization for A100 GPU
- QLoRA configuration optimized for A100's 40GB VRAM
- Batch size of 2 with gradient accumulation (effective batch size = 8)
- 4-bit quantization reduces memory footprint by ~75%
- Gradient checkpointing enabled for additional memory savings

### Training Tips
- Start with a small dataset to test your setup
- Monitor GPU memory usage during training
- Save checkpoints regularly to avoid losing progress

## Troubleshooting

### Common Issues

1. **GPU Memory Issues on A100**: 
   - Reduce batch size from 2 to 1 if needed
   - Increase gradient accumulation steps to maintain effective batch size
   - Use smaller sequence lengths for longer documents

2. **Installation Issues in Colab**:
   - Restart runtime if package conflicts occur
   - Use `!pip install` for compatibility in Colab environment
   - Ensure GPU runtime is selected (Runtime → Change runtime type → A100 GPU)

3. **Model Loading Issues**:
   - Ensure you have Hugging Face authentication set up
   - Check internet connection for model downloads
   - Verify sufficient disk space

## Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [UV Documentation](https://docs.astral.sh/uv/)

## Acknowledgments

- Meta AI for the Llama3 model
- Hugging Face for the transformers and PEFT libraries
- The QLoRA research team for the quantization techniques
- Astral for the UV package manager