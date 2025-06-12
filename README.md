# 🚀 Run LLMs on Modal with vLLM!
---
**Made with AI (yes I vibe coded this but dont worry I tested my code)**

[![GitHub stars](https://img.shields.io/github/stars/realblehhguh/modal-vllm-server?style=social)](https://github.com/yourusername/your-repo-name)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://python.org)

A sophisticated vLLM-based OpenAI-compatible API server that automatically selects appropriate GPU configurations and optimizes settings based on the model being served. 🎯
---
## ✨ Features

- 🧠 **Dynamic GPU Selection**: Automatically chooses the right GPU (T4, A10G, A100, H100) based on model size and type
- ⚡ **Automatic Model Optimization**: Configures vLLM parameters optimally for each model
- 🗜️ **Quantization Support**: Handles GPTQ, AWQ, and other quantized models automatically  
- ☁️ **Modal Integration**: Seamless deployment on Modal's serverless GPU infrastructure
- 📦 **Model Management**: Easy model downloading and caching
- 🔄 **OpenAI Compatible**: Drop-in replacement for OpenAI API

## 🚀 Quick Start

### 📋 Prerequisites

- [Modal](https://modal.com) account and CLI installed 🌐
- Hugging Face account for model access 🤗
- Python 3.11+ 🐍

### ⚙️ Setup

1. **Install Modal CLI** 💻:
   ```bash
   pip install modal
   modal setup
   ```

2. **Set up Hugging Face token** 🔑:
   ```bash
   modal secret create huggingface HF_TOKEN=your_hf_token_here
   ```

3. **Clone and run** 🏃‍♂️:
   ```bash
   git clone https://github.com/realblehhguh/modal-vllm-server.git
   cd modal-vllm-server
   modal run vllmserver.py::chat
   ```

## 💡 Usage Examples

### 💬 Basic Chat Session
```bash
# Use default model (DialoGPT-medium)
modal run vllmserver.py::chat

# Use specific model
MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0' modal run vllmserver.py::chat
```

### ❓ Custom Questions
```bash
modal run vllmserver.py::chat --questions "Hello!|What can you do?|Write Python code|Thanks!"
```

### 📥 Pre-download Models
```bash
modal run vllmserver.py::download --model-name "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
```

### 🧪 Test Specific Models
```bash
modal run vllmserver.py::test_llama31
```

## 🤖 Supported Models

The system automatically detects and optimizes for:

- 🦙 **Large Models (7B-70B)**: Llama, Mistral, CodeLlama
- 🗜️ **Quantized Models**: GPTQ, AWQ, INT4/INT8 variants
- 🐣 **Small Models (1-2B)**: TinyLlama, small chat models
- 💻 **Code Models**: CodeLlama, StarCoder variants

### 🎯 GPU Selection Logic

| Model Size | Quantized | GPU Selected | Memory Utilization |
|------------|-----------|--------------|-------------------|
| 70B | ❌ | H100 🚀 | 90% |
| 70B | ✅ | A100 ⚡ | 85% |
| 7B-13B | ❌ | A100 ⚡ | 85% |
| 7B-13B | ✅ | A10G 💪 | 80% |
| 3B-6B | Any | A10G 💪 | 80% |
| 1B-2B | Any | T4 🔧 | 80% |

## ⚙️ Configuration

### 🔧 Environment Variables

- `MODEL_NAME`: Hugging Face model identifier 🤗
- `HF_TOKEN`: Hugging Face access token (set via Modal secrets) 🔐

### 🎛️ vLLM Configuration

The system automatically configures:
- `max_model_len`: Context length based on model and GPU 📏
- `tensor_parallel_size`: GPU parallelization 🔀
- `quantization`: Detected from model name 🗜️
- `dtype`: Optimal data type for GPU 📊
- `gpu_memory_utilization`: Safe memory usage 💾

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐
│ 👤 User Request │───▶│ 🎯 GPU Selection │───▶│ 🚀 vLLM Server│
│                 │    │   Logic          │    │               │
└─────────────────┘    └──────────────────┘    └───────────────┘
                              │                        │
                              ▼                        ▼
                       ┌──────────────────┐    ┌───────────────┐
                       │ 📥 Model Download│    │ 🔌 OpenAI API │
                       │ & Optimization   │    │ Endpoint      │
                       └──────────────────┘    └───────────────┘
```

## 🚀 Advanced Features

### 🔍 Model Integrity Checking
- ✅ Validates essential model files on startup
- 🔄 Automatic re-download if corruption detected
- ⏯️ Resume interrupted downloads

### ⚡ Performance Optimizations
- 🧠 Prefix caching for repeated patterns
- 📝 Chunked prefill for large contexts
- 🎯 Optimized batch processing
- 💾 Memory-efficient tensor parallelization

### 🛡️ Error Handling
- 🔬 Comprehensive startup diagnostics
- 🔧 GPU compatibility checking
- 🔄 Graceful fallback options
- 📊 Detailed error reporting

## 🔌 API Usage

Once running, the server provides OpenAI-compatible endpoints:

### 🐍 Python Example
```python
import openai

client = openai.OpenAI(
    base_url="http://your-modal-url/v1",
    api_key="vllm"  # Can be any string
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello! 👋"}],
    max_tokens=100
)
```

### 🌐 Curl Examples

#### 💬 Chat Completions
```bash
# Basic chat completion
curl -X POST "https://your-modal-url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vllm" \
  -d '{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}'
```

#### 🤖 Multi-turn Conversation
```bash
curl -X POST "https://your-modal-url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vllm" \
  -d '{
  "model": "your-model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms"},
    {"role": "assistant", "content": "Quantum computing uses quantum bits..."},
    {"role": "user", "content": "How is it different from classical computing?"}
  ],
  "max_tokens": 200,
  "temperature": 0.5
}'
```

#### 💻 Code Generation
```bash
curl -X POST "https://your-modal-url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vllm" \
  -d '{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
  ],
  "max_tokens": 300,
  "temperature": 0.2,
  "stop": ["```"]
}'
```

#### 🔄 Streaming Response
```bash
curl -X POST "https://your-modal-url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vllm" \
  -d '{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "Tell me a short story about a robot"}
  ],
  "max_tokens": 150,
  "stream": true
}'
```

#### 📊 Model Information
```bash
# List available models
curl -H "Authorization: Bearer vllm" \
  "https://your-modal-url/v1/models"
```

#### 🎛️ Advanced Parameters
```bash
curl -X POST "https://your-modal-url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vllm" \
  -d '{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "Generate creative marketing copy for a coffee shop"}
  ],
  "max_tokens": 250,
  "temperature": 0.8,
  "top_p": 0.9,
  "frequency_penalty": 0.1,
  "presence_penalty": 0.1,
  "stop": ["\n\n"]
}'
```

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✨ Make your changes
4. 🧪 Add tests if applicable
5. 📋 Submit a pull request

## 🔧 Troubleshooting

### ⚠️ Common Issues

**💥 Out of Memory Errors**:
- 📉 Try a smaller model or quantized version
- 📏 Reduce `max_model_len` in configuration
- 🆙 Use a larger GPU tier

**📥 Model Download Failures**:
- 🔑 Check HF_TOKEN permissions
- ✏️ Verify model name spelling
- 💾 Ensure sufficient disk space

**⏰ Startup Timeouts**:
- ⏳ Large models take 5-8 minutes to load
- 🔍 Check GPU availability
- 📊 Monitor Modal logs for progress

## 📈 Changelog

### v1.0.0 🎉
- 🚀 Initial release with vLLM 0.9.1
- 🎯 Dynamic GPU selection
- ⚡ Automatic model optimization
- 🔌 OpenAI API compatibility

## 🙏 Acknowledgements

This project stands on the shoulders of giants. Special thanks to:

- **[vLLM Team](https://github.com/vllm-project/vllm)** 🚀 - For creating the incredible vLLM inference engine that powers this server
- **[Modal Labs](https://modal.com)** ☁️ - For providing the serverless GPU infrastructure that makes this possible
- **[Hugging Face](https://huggingface.co)** 🤗 - For hosting the vast ecosystem of open-source models
- **[OpenAI](https://openai.com)** 🧠 - For establishing the API standards that ensure compatibility
- **The Open Source AI Community** 🌟 - For developing and sharing the amazing models that make this all worthwhile

### 🔧 Technologies Used
- **[Python](https://python.org)** - The backbone language
- **[FastAPI](https://fastapi.tiangolo.com)** - For the robust API framework
- **[PyTorch](https://pytorch.org)** - The deep learning foundation
- **[Transformers](https://github.com/huggingface/transformers)** - For model handling and tokenization

### 💡 Inspiration
This project was inspired by the need for a simple, automated way to deploy LLMs without worrying about GPU selection and optimization details. The goal was to make powerful language models as accessible as possible while maintaining production-ready performance.

---

**Built with ❤️ by the community, for the community** 🌍
