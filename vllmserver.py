import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import modal

# --- Configuration ---
DEFAULT_MODEL = "microsoft/DialoGPT-medium"

# --- Modal App Setup ---
app = modal.App("vllm-openai-server")

# --- Helper functions ---
def get_model_info(model_name: str):
    """Get model-specific paths and volume name"""
    volume_name = f"vllm-model-vol-{model_name.replace('/', '--').replace('_', '-')}"
    model_base_path = Path("/model")
    model_path = model_base_path / model_name.split("/")[-1]
    return volume_name, model_base_path, model_path

def download_model_to_path(model_name: str, model_path: Path):
    """Download model to specific path"""
    import os
    import shutil
    from huggingface_hub import snapshot_download
    
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if model_path.exists():
        print(f"Model directory exists at {model_path}, checking integrity...")
        
        # Check for essential files
        essential_files = ["config.json"]
        
        missing_essential = []
        for file in essential_files:
            file_path = model_path / file
            if not file_path.exists() or file_path.stat().st_size == 0:
                missing_essential.append(file)
        
        if missing_essential:
            print(f"Found missing essential files: {missing_essential}")
            print("Removing corrupted model directory...")
            shutil.rmtree(model_path)
        else:
            print("Model integrity check passed.")

    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"📥 Downloading {model_name} to {model_path}...")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN"),
                resume_download=True,
                force_download=False,
            )
            print("✅ Download complete.")
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if model_path.exists():
                shutil.rmtree(model_path)
            raise
    else:
        print(f"✅ Model already exists at {model_path}")

# --- Dynamic GPU selection based on model ---
def get_gpu_config(model_name: str):
    """Get appropriate GPU configuration based on model size"""
    model_name_lower = model_name.lower()
    
    # Large models (7B+ parameters) - check for quantized versions
    if any(size in model_name_lower for size in ["7b", "8b", "13b", "70b"]):
        # Quantized models need less memory
        if any(quant in model_name_lower for quant in ["gptq", "int4", "int8", "awq", "gguf"]):
            if "70b" in model_name_lower:
                return "A100", 0.85  # 70B quantized can fit on A100
            elif any(size in model_name_lower for size in ["13b", "7b", "8b"]):
                return "A10G", 0.8  # 7B-13B quantized can fit on A10G
        else:
            # Full precision models
            if "70b" in model_name_lower:
                return "H100", 0.9
            elif any(size in model_name_lower for size in ["13b", "7b", "8b"]):
                return "A100", 0.85
    
    # Small models (1-2B parameters)
    elif any(size in model_name_lower for size in ["1b", "2b", "mini", "small", "tinyllama"]):
        return "T4", 0.8
    
    # Medium models (3-6B parameters)  
    elif any(size in model_name_lower for size in ["3b", "4b", "6b"]):
        return "A10G", 0.8
    
    # Default for unknown sizes
    else:
        return "A10G", 0.8

# --- Dynamic vLLM configuration based on model ---
def get_vllm_config(model_name: str, gpu_type: str):
    """Get vLLM configuration based on model and GPU"""
    config = {
        "max_model_len": 2048,
        "max_num_seqs": 4,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "quantization": None,  # Will be set if quantized model detected
    }
    
    model_lower = model_name.lower()
    
    # Detect quantization
    if "gptq" in model_lower:
        config["quantization"] = "gptq"
    elif "awq" in model_lower:
        config["quantization"] = "awq"
    elif "int4" in model_lower:
        config["quantization"] = "gptq"  # INT4 usually means GPTQ
    elif "int8" in model_lower:
        config["quantization"] = "gptq"
    elif "gguf" in model_lower:
        config["quantization"] = None  # GGUF handled differently in vLLM 0.9+
    
    # Adjust based on model size
    if "70b" in model_lower:
        config["max_model_len"] = 4096 if config["quantization"] else 2048
        config["max_num_seqs"] = 8 if config["quantization"] else 4
        config["tensor_parallel_size"] = 2 if gpu_type in ["H100", "A100"] else 1
    elif any(size in model_lower for size in ["7b", "8b", "13b"]):
        config["max_model_len"] = 8192 if config["quantization"] else 4096
        config["max_num_seqs"] = 16 if config["quantization"] else 8
    elif any(size in model_lower for size in ["3b", "4b", "6b"]):
        config["max_model_len"] = 6144
        config["max_num_seqs"] = 12
    elif any(size in model_lower for size in ["1b", "2b", "tinyllama"]):
        config["max_model_len"] = 8192
        config["max_num_seqs"] = 16
    
    # Adjust based on GPU capabilities
    if gpu_type == "T4":
        config["max_model_len"] = min(config["max_model_len"], 4096)
        config["max_num_seqs"] = min(config["max_num_seqs"], 8)
        config["dtype"] = "half"  # Force float16 for T4 compatibility
    elif gpu_type in ["A100", "H100"]:
        config["dtype"] = "auto"  # These support bfloat16
    else:  # A10G and others
        config["dtype"] = "auto"  # vLLM 0.9+ handles dtype selection better
    
    return config

# --- Updated container image with vLLM 0.9.1 ---
base_image = (
    modal.Image.debian_slim(python_version="3.11")  # Updated to Python 3.11
    .apt_install("curl", "git")
    .pip_install(
        "numpy<2.0",
        "vllm==0.9.1",  # Updated to v0.9.1
        "transformers==4.46.2",  # Latest transformers
        "torch==2.5.1",  # Latest PyTorch
        "accelerate==1.1.1",  # Updated accelerate
        "openai==1.54.4",  # Latest OpenAI client
        "huggingface_hub[hf_transfer]",
        "tokenizers==0.20.3",  # Latest tokenizers
        "requests",
        "auto-gptq>=0.7.1",  # Updated GPTQ support
        "autoawq>=0.2.6",    # Updated AWQ support
        "optimum>=1.23.0",   # Latest optimum
        "scipy",             # Sometimes needed for quantized models
        "sentencepiece",     # For some tokenizers
        "protobuf",          # For some model formats
    )
)

# --- Core chat function ---
def run_chat_logic(model_name: str, custom_questions: Optional[list] = None):
    """Core chat logic that runs the vLLM server and handles chat"""
    # Get model configuration
    gpu_type, gpu_memory_util = get_gpu_config(model_name)
    volume_name, model_base_path, model_path = get_model_info(model_name)
    
    port = 8000
    
    print(f"🤖 Using model: {model_name}")
    print(f"🔧 GPU: {gpu_type}, Memory utilization: {gpu_memory_util}")
    print(f"📁 Model path: {model_path}")
    
    # Download model if it doesn't exist
    if not model_path.exists():
        print(f"📥 Model not found at {model_path}, downloading...")
        download_model_to_path(model_name, model_path)
    
    if not model_path.exists():
        raise RuntimeError(f"Model path {model_path} does not exist after download")
    
    # Show model files
    files = list(model_path.glob("*"))
    print(f"📁 Model directory contains {len(files)} files")
    essential_files = ["config.json", "pytorch_model.bin", "model.safetensors", "quantize_config.json"]
    for efile in essential_files:
        epath = model_path / efile
        if epath.exists():
            print(f"   ✅ {efile} ({epath.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"   ❌ {efile} (missing)")
    
    # Get dynamic configuration
    vllm_config = get_vllm_config(model_name, gpu_type)
    
    # Build vLLM command with GPU-specific settings
    vllm_command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--model", str(model_path),
        "--served-model-name", model_name,
        "--tensor-parallel-size", str(vllm_config["tensor_parallel_size"]),
        "--trust-remote-code",
        "--max-model-len", str(min(vllm_config["max_model_len"], 4096)),  # Less conservative with 0.9.1
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--max-num-batched-tokens", str(min(vllm_config["max_num_seqs"] * 512, 8192)),  # New parameter in 0.9+
        "--disable-log-requests",
        "--tokenizer-mode", "auto",
        "--dtype", vllm_config["dtype"],
        "--enable-prefix-caching",  # New optimization in 0.9+
    ]
    
    # Add quantization if detected
    if vllm_config["quantization"]:
        vllm_command.extend(["--quantization", vllm_config["quantization"]])
        print(f"🔧 Detected quantization: {vllm_config['quantization']}")
    
    # vLLM 0.9.1 specific optimizations
    if gpu_type in ["A100", "H100"]:
        vllm_command.append("--enable-chunked-prefill")  # Better for large contexts
    
    print("🚀 Starting vLLM server...")
    print(f"⚙️ Config: max_len={min(vllm_config['max_model_len'], 4096)}, batched_tokens={min(vllm_config['max_num_seqs'] * 512, 8192)}, dtype={vllm_config['dtype']}")
    if vllm_config["quantization"]:
        print(f"🗜️ Quantization: {vllm_config['quantization']}")
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env["VLLM_USE_MODELSCOPE"] = "False"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    vllm_process = subprocess.Popen(
        vllm_command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        env=env
    )
    
    # Wait for startup with better monitoring
    max_retries = 240  # 8 minutes for larger models
    output_lines = []
    
    for i in range(max_retries):
        # Check if process is still running
        if vllm_process.poll() is not None:
            # Process died, collect all output
            try:
                remaining_output = vllm_process.stdout.read()
                if remaining_output:
                    output_lines.extend(remaining_output.split('\n'))
            except:
                pass
            
            print(f"❌ vLLM process terminated with code: {vllm_process.returncode}")
            print(f"📝 Process output (last 30 lines):")
            for line in output_lines[-30:]:
                if line.strip():
                    print(f"   {line.strip()}")
            
            # Provide helpful error hints
            output_text = '\n'.join(output_lines)
            if "rope_scaling" in output_text:
                print("\n💡 Hint: Model configuration issue - vLLM 0.9.1 should handle this better")
            elif "out of memory" in output_text.lower():
                print("\n💡 Hint: Try reducing max_model_len or using a larger GPU")
                print("   Or reduce gpu_memory_utilization")
            elif "quantization" in output_text.lower():
                print("\n💡 Hint: Check if the quantization format is supported in vLLM 0.9.1")
            elif "cuda" in output_text.lower() and "error" in output_text.lower():
                print("\n💡 Hint: CUDA compatibility issue - check GPU driver support")
            
            raise RuntimeError(f"vLLM failed to start (exit code: {vllm_process.returncode})")
        
        # Read any new output
        try:
            line = vllm_process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                if i % 30 == 0 and line.strip():  # Print some output periodically
                    print(f"   {line.strip()}")
        except:
            pass
        
        # Test if server is ready
        try:
            result = subprocess.run(
                ["curl", "-f", f"http://localhost:{port}/health"],
                check=True, capture_output=True, timeout=5
            )
            print("✅ vLLM server is ready!")
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if i % 60 == 0:  # Progress update every 2 minutes
                print(f"⏳ Waiting for server... ({i+1}/{max_retries}) - Large models take time to load")
            time.sleep(2)
    else:
        # Timeout reached
        print("⏰ Startup timeout reached!")
        print(f"📝 Recent output:")
        for line in output_lines[-20:]:
            if line.strip():
                print(f"   {line.strip()}")
        
        vllm_process.terminate()
        raise RuntimeError("vLLM server failed to start within timeout period")
    
    # Continue with chat logic...
    with modal.forward(port) as tunnel:
        print(f"🌐 Server URL: {tunnel.url}")
        
        # Simple test first
        import requests
        try:
            test_response = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
            print(f"✅ API test successful: {test_response.status_code}")
            
            # Also test completions endpoint
            models_data = test_response.json()
            print(f"📋 Available model: {models_data.get('data', [{}])[0].get('id', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  API test failed: {e}")
        
        # Use custom questions or defaults
        if custom_questions:
            questions = custom_questions
        else:
            # Model-specific default questions  
            if "llama" in model_name.lower() and any(v in model_name.lower() for v in ["3.1", "3-1"]):
                questions = [
                    "Hello! I'm testing Llama 3.1. Please introduce yourself briefly.",
                    "What are your key capabilities as an AI assistant?",
                    "Can you write a simple Python function to calculate fibonacci numbers?",
                    "What's the difference between lists and tuples in Python?",
                    "Thanks for the demonstration!"
                ]
            elif "tinyllama" in model_name.lower():
                questions = [
                    "Hello! What's your name?",
                    "Can you help me with math: what's 5 + 3?",
                    "Tell me a short joke",
                    "Thanks!"
                ]
            elif "code" in model_name.lower() or "coder" in model_name.lower():
                questions = [
                    "Write a Python function to reverse a string",
                    "How do I sort a list in Python?", 
                    "What's the difference between == and is in Python?",
                    "Thanks for the help!"
                ]
            else:
                questions = [
                    "Hello! Please introduce yourself briefly.",
                    "What can you help me with today?",
                    "Thank you for the demo!"
                ]
        
        conversation = []
        
        print(f"\n{'='*60}")
        print(f"🤖 Chat Session with {model_name}")
        print(f"{'='*60}\n")
        
        for question in questions:
            print(f"👤 You: {question}")
            
            conversation.append({"role": "user", "content": question})
            
            try:
                response = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json", 
                        "Authorization": "Bearer vllm"
                    },
                    json={
                        "model": model_name,
                        "messages": conversation,
                        "max_tokens": 300,  # Increased for better responses
                        "temperature": 0.8,
                        "top_p": 0.9,
                    },
                    timeout=90
                )
                
                if response.status_code == 200:
                    ai_response = response.json()["choices"][0]["message"]["content"]
                    conversation.append({"role": "assistant", "content": ai_response})
                    print(f"🤖 AI: {ai_response}\n")
                else:
                    print(f"❌ Error: {response.status_code} - {response.text}\n")
                    
            except Exception as e:
                print(f"❌ Error: {e}\n")
            
            time.sleep(1)
        
        print("✅ Chat session completed!")
        print(f"🌐 Server is still running at: {tunnel.url}")
        
        print("\n⏰ Keeping server alive for 5 minutes for additional testing...")
        print("   You can use the URL above to make API calls")
        try:
            time.sleep(300)
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
        finally:
            vllm_process.terminate()

# --- GPU-specific chat functions ---
@app.function(
    gpu="T4",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_t4(model_name: str, custom_questions: Optional[list] = None):
    """Run chat session on T4 GPU"""
    return run_chat_logic(model_name, custom_questions)

@app.function(
    gpu="A10G",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_a10g(model_name: str, custom_questions: Optional[list] = None):
    """Run chat session on A10G GPU"""
    return run_chat_logic(model_name, custom_questions)

@app.function(
    gpu="A100",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_a100(model_name: str, custom_questions: Optional[list] = None):
    """Run chat session on A100 GPU"""
    return run_chat_logic(model_name, custom_questions)

@app.function(
    gpu="H100",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_h100(model_name: str, custom_questions: Optional[list] = None):
    """Run chat session on H100 GPU"""
    return run_chat_logic(model_name, custom_questions)

# --- Model management functions ---
@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def download_model_remote(model_name: str):
    """Download a model remotely"""
    volume_name, model_base_path, model_path = get_model_info(model_name)
    
    print(f"📥 Downloading model: {model_name}")
    download_model_to_path(model_name, model_path)
    
    return f"✅ Model {model_name} downloaded successfully"

# --- Helper function to get the right chat function ---
def get_chat_function(model_name: str):
    """Get the appropriate chat function based on model requirements"""
    gpu_type, _ = get_gpu_config(model_name)
    
    if gpu_type == "T4":
        return run_chat_t4
    elif gpu_type == "A10G":
        return run_chat_a10g
    elif gpu_type == "A100":
        return run_chat_a100
    elif gpu_type == "H100":
        return run_chat_h100
    else:
        return run_chat_a10g

# --- Local entrypoints ---
@app.local_entrypoint()
def chat(questions: str = ""):
    """Start chat session with current model"""
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🚀 Starting chat session...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    
    # Parse custom questions if provided
    custom_questions = None
    if questions:
        custom_questions = [q.strip() for q in questions.split("|") if q.strip()]
        print(f"📝 Using {len(custom_questions)} custom questions")
    
    # Get the appropriate chat function based on GPU requirements
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions)

@app.local_entrypoint()
def download(model_name: str):
    """Download a specific model"""
    if not model_name:
        print("❌ Please specify a model name")
        return
    
    print(f"📥 Downloading model: {model_name}")
    try:
        result = download_model_remote.remote(model_name)
        print(result)
        print(f"💡 To use this model: MODEL_NAME='{model_name}' modal run script.py::chat")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")

@app.local_entrypoint()
def info():
    """Show current configuration"""
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, gpu_util = get_gpu_config(current_model)
    vllm_config = get_vllm_config(current_model, gpu_type)
    
    print(f"🚀 vLLM v0.9.1 Configuration:")
    print(f"  Model: {current_model}")
    print(f"  GPU: {gpu_type} ({gpu_util*100}% memory)")
    print(f"  Max length: {vllm_config['max_model_len']}")
    print(f"  Max batched tokens: {min(vllm_config['max_num_seqs'] * 512, 8192)}")
    print(f"  Tensor parallel: {vllm_config['tensor_parallel_size']}")
    print(f"  Dtype: {vllm_config['dtype']}")
    if vllm_config['quantization']:
        print(f"  Quantization: {vllm_config['quantization']}")
    
    print(f"\n📋 Usage Examples:")
    print(f"  Llama 3.1 8B GPTQ: MODEL_NAME='hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4' modal run script.py::chat")
    print(f"  TinyLlama: MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0' modal run script.py::chat")

# --- Test functions ---
@app.local_entrypoint()
def test_llama31():
    """Test Llama 3.1 8B GPTQ with vLLM 0.9.1"""
    model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    print(f"🧪 Testing {model_name} with vLLM 0.9.1")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! I'm testing Llama 3.1 with vLLM 0.9.1. Please introduce yourself.",
        "What new features does Llama 3.1 bring?",
        "Write a Python function to calculate the factorial of a number using recursion.",
        "Explain the concept of machine learning in simple terms.",
        "Thanks for the comprehensive test!"
    ])

@app.local_entrypoint()
def version():
    """Show vLLM version info"""
    print("📦 Package Versions:")
    print("  vLLM: 0.9.1")
    print("  Transformers: 4.46.2")
    print("  PyTorch: 2.5.1")
    print("  Python: 3.11")
