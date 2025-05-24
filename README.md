# Llama3-8B Fine-tuning with QLoRA

## About

This project demonstrates how to fine-tune Meta's Llama3-8B model using QLoRA (Quantized Low-Rank Adaptation), an efficient parameter-efficient fine-tuning technique. QLoRA allows you to fine-tune large language models on consumer hardware by using 4-bit quantization and low-rank adapters, significantly reducing memory requirements while maintaining performance.

### Key Features
- **Memory Efficient**: Uses 4-bit quantization to reduce memory footprint
- **Parameter Efficient**: QLoRA adapters require minimal additional parameters
- **High Performance**: Maintains model quality while being computationally efficient
- **Consumer Hardware Friendly**: Can run on GPUs with limited VRAM

### What You'll Learn
- How to set up a QLoRA fine-tuning pipeline
- Memory optimization techniques for large language models
- Parameter-efficient fine-tuning strategies
- Best practices for working with Llama3-8B

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- Git
- [UV package manager](https://github.com/astral-sh/uv) (recommended for fast dependency management)

## Installation Guide

### Option 1: Using UV (Recommended)

UV is a fast Python package manager that can significantly speed up dependency installation.

#### 1. Install UV
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
git clone https://github.com/your-username/Llama3-8b-Finetuning-QLoRA.git
cd Llama3-8b-Finetuning-QLoRA
```

#### 3. Initialize Virtual Environment with UV
```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### 4. Install Dependencies with UV
```bash
# Install all required packages using UV for faster installation
uv pip install peft
uv pip install accelerate  
uv pip install bitsandbytes
uv pip install datasets
uv pip install evaluate
uv pip install transformers
uv pip install torch

# Or install additional packages as needed
uv pip install jupyter ipykernel
```

### Option 2: Using Standard Pip

If you prefer using standard pip or don't have UV installed:

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Llama3-8b-Finetuning-QLoRA.git
cd Llama3-8b-Finetuning-QLoRA
```

#### 2. Create Virtual Environment
```bash
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install peft accelerate bitsandbytes datasets evaluate transformers torch jupyter ipykernel
```

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

## Usage Tips

### Using UV in Jupyter Notebooks
- **For faster package installation**: Use `!uv pip install package_name` instead of `!pip install package_name`
- **For sharing**: When sharing notebooks, use standard `pip install` commands for compatibility
- **For development**: UV provides much faster dependency resolution and installation

### Memory Optimization
- Use the QLoRA configuration in the notebook for memory efficiency
- Adjust batch sizes based on your GPU memory
- Consider gradient checkpointing for even lower memory usage

### Training Tips
- Start with a small dataset to test your setup
- Monitor GPU memory usage during training
- Save checkpoints regularly to avoid losing progress

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: 
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller sequence lengths

2. **Installation Issues**:
   - Ensure CUDA drivers are properly installed
   - Try installing PyTorch with specific CUDA version
   - Use UV for faster dependency resolution

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