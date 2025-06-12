import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import json

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

def get_model_config(model_path: Path):
    """Read model config to get actual limits"""
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract key parameters
            max_position_embeddings = config.get("max_position_embeddings", None)
            model_max_length = config.get("model_max_length", None) or config.get("max_sequence_length", None)
            vocab_size = config.get("vocab_size", None)
            hidden_size = config.get("hidden_size", None)
            model_type = config.get("model_type", "unknown")
            architectures = config.get("architectures", [])
            
            print(f"📖 Model config: type={model_type}, arch={architectures}, max_pos_emb={max_position_embeddings}, max_len={model_max_length}")
            
            return {
                "max_position_embeddings": max_position_embeddings,
                "model_max_length": model_max_length,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "model_type": model_type,
                "architectures": architectures
            }
        except Exception as e:
            print(f"⚠️ Could not read model config: {e}")
            return None
    return None

def is_model_supported(model_config: dict, model_name: str) -> bool:
    """Check if the model is likely supported by vLLM"""
    if not model_config:
        return True  # Assume supported if we can't read config
    
    model_type = model_config.get("model_type", "").lower()
    architectures = [arch.lower() for arch in model_config.get("architectures", [])]
    
    # Known problematic model types for vLLM 0.9.1
    problematic_types = ["stablelm", "stablelmepoch"]
    problematic_architectures = ["stablelmlayer", "stablelm"]
    
    if model_type in problematic_types:
        print(f"⚠️ Model type '{model_type}' may have compatibility issues with vLLM 0.9.1")
        return False
    
    if any(arch in problematic_architectures for arch in architectures):
        print(f"⚠️ Architecture {architectures} may have compatibility issues with vLLM 0.9.1")
        return False
    
    return True

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
                token=os.environ.get("HF_TOKEN"),
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
    
    # Massive models (70B+ parameters)
    if any(size in model_name_lower for size in ["70b", "72b", "405b"]):
        if any(quant in model_name_lower for quant in ["gptq", "int4", "int8", "awq"]):
            if "405b" in model_name_lower:
                return "B200", 0.90  # 405B quantized needs B200
            elif any(size in model_name_lower for size in ["70b", "72b"]):
                return "H100", 0.85  # 70B quantized can fit on H100
        else:
            # Full precision massive models
            if "405b" in model_name_lower:
                return "B200", 0.95  # 405B full precision needs B200
            elif any(size in model_name_lower for size in ["70b", "72b"]):
                return "A100-80GB", 0.90  # 70B full precision needs A100-80GB
    
    # Large models (7B-34B parameters)
    elif any(size in model_name_lower for size in ["7b", "8b", "13b", "14b", "34b"]):
        if any(quant in model_name_lower for quant in ["gptq", "int4", "int8", "awq"]):
            if any(size in model_name_lower for size in ["34b"]):
                return "A100-40GB", 0.85  # 34B quantized needs A100-40GB
            elif any(size in model_name_lower for size in ["13b", "14b"]):
                return "A10G", 0.80  # 13B quantized fits on A10G
            elif any(size in model_name_lower for size in ["7b", "8b"]):
                return "L4", 0.75   # 7B-8B quantized can fit on L4
        else:
            # Full precision large models
            if any(size in model_name_lower for size in ["34b"]):
                return "A100-80GB", 0.85  # 34B full precision needs A100-80GB
            elif any(size in model_name_lower for size in ["13b", "14b"]):
                return "A100-40GB", 0.80  # 13B full precision needs A100-40GB
            elif any(size in model_name_lower for size in ["7b", "8b"]):
                return "L40S", 0.80  # 7B-8B full precision works well on L40S
    
    # Small models (1-2B parameters)
    elif any(size in model_name_lower for size in ["1b", "2b", "mini", "small", "tinyllama"]):
        return "T4", 0.8
    
    # Medium models (3-6B parameters)  
    elif any(size in model_name_lower for size in ["3b", "4b", "6b"]):
        if "stablelm" in model_name_lower or "stablecode" in model_name_lower:
            return "A10G", 0.8  # StableLM models need more resources
        elif any(size in model_name_lower for size in ["6b"]):
            return "L40S", 0.75  # 6B models work well on L40S
        elif any(size in model_name_lower for size in ["3b", "4b"]):
            return "L4", 0.75
        else:
            return "A10G", 0.8
    
    # Chat/dialog models (usually medium sized)
    elif any(term in model_name_lower for term in ["dialog", "chat", "gpt2", "dialo"]):
        return "L4", 0.75  # Most chat models are small-medium
    
    # Default for unknown sizes
    else:
        return "L4", 0.75  # Conservative default

# --- Dynamic vLLM configuration based on model ---
def get_vllm_config(model_name: str, gpu_type: str, model_config: dict = None):
    """Get vLLM configuration based on model and GPU"""
    config = {
        "max_model_len": 2048,
        "max_num_seqs": 4,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "quantization": None,
        "trust_remote_code": True,
    }
    
    model_lower = model_name.lower()
    
    # Use actual model config if available
    if model_config:
        max_pos_emb = model_config.get("max_position_embeddings")
        model_max_len = model_config.get("model_max_length")
        
        if max_pos_emb:
            config["max_model_len"] = int(max_pos_emb * 0.9)
        elif model_max_len:
            config["max_model_len"] = min(model_max_len, 2048)
        
        print(f"🔧 Adjusted max_model_len to {config['max_model_len']} based on model config")
    
    # Detect quantization
    if "gptq" in model_lower:
        config["quantization"] = "gptq"
    elif "awq" in model_lower:
        config["quantization"] = "awq"
    elif "int4" in model_lower:
        config["quantization"] = "gptq"
    elif "int8" in model_lower:
        config["quantization"] = "gptq"
    elif "gguf" in model_lower:
        config["quantization"] = None
    
    # Model size based adjustments
    if any(size in model_lower for size in ["405b"]):
        config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 16 if config["quantization"] else 8
        config["tensor_parallel_size"] = 8 if gpu_type == "B200" else 4
    elif any(size in model_lower for size in ["70b", "72b"]):
        config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 12 if config["quantization"] else 6
        config["tensor_parallel_size"] = 4 if gpu_type in ["H100", "H200", "A100-80GB", "B200"] else 2
    elif any(size in model_lower for size in ["34b"]):
        config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 10 if config["quantization"] else 6
        config["tensor_parallel_size"] = 2 if gpu_type in ["A100-40GB", "A100-80GB", "H100", "H200", "L40S"] else 1
    elif any(size in model_lower for size in ["7b", "8b", "13b", "14b"]):
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 16 if config["quantization"] else 8
        config["tensor_parallel_size"] = 2 if gpu_type in ["A100-40GB", "A100-80GB", "H100", "H200", "L40S"] else 1
    elif any(size in model_lower for size in ["3b", "4b", "6b"]):
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 6144)
        config["max_num_seqs"] = 12
    elif any(size in model_lower for size in ["1b", "2b", "tinyllama"]):
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 8192)
        config["max_num_seqs"] = 16
    elif "dialog" in model_lower or "chat" in model_lower:
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 1024)
        config["max_num_seqs"] = 8
    
    # Model-specific adjustments
    if "stablelm" in model_lower:
        config["max_model_len"] = min(config["max_model_len"], 2048)
        config["max_num_seqs"] = 4
        config["dtype"] = "half"
        config["trust_remote_code"] = True
    
    # GPU-specific adjustments
    if gpu_type == "T4":
        config["max_model_len"] = min(config["max_model_len"], 2048)
        config["max_num_seqs"] = min(config["max_num_seqs"], 8)
        config["dtype"] = "half"
    elif gpu_type == "L4":
        config["max_model_len"] = min(config["max_model_len"], 4096)
        config["max_num_seqs"] = min(config["max_num_seqs"], 12)
        config["dtype"] = "half"
    elif gpu_type == "A10G":
        config["max_model_len"] = min(config["max_model_len"], 6144)
        config["max_num_seqs"] = min(config["max_num_seqs"], 16)
        config["dtype"] = "auto"
    elif gpu_type == "L40S":
        config["max_model_len"] = min(config["max_model_len"], 8192)
        config["max_num_seqs"] = min(config["max_num_seqs"], 20)
        config["dtype"] = "auto"
    elif gpu_type == "A100-40GB":
        config["max_model_len"] = min(config["max_model_len"], 8192)
        config["max_num_seqs"] = min(config["max_num_seqs"], 24)
        config["dtype"] = "auto"
    elif gpu_type == "A100-80GB":
        config["max_model_len"] = min(config["max_model_len"], 16384)
        config["max_num_seqs"] = min(config["max_num_seqs"], 32)
        config["dtype"] = "auto"
    elif gpu_type in ["H100", "H200"]:
        config["max_model_len"] = min(config["max_model_len"], 16384)
        config["max_num_seqs"] = min(config["max_num_seqs"], 48)
        config["dtype"] = "auto"
    elif gpu_type == "B200":
        config["max_model_len"] = min(config["max_model_len"], 32768)
        config["max_num_seqs"] = min(config["max_num_seqs"], 64)
        config["dtype"] = "auto"
    
    # Ensure minimum viable values
    config["max_model_len"] = max(config["max_model_len"], 256)
    
    return config

# --- Container image with clean dependency resolution ---
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git")
    .pip_install(
        "torch==2.5.1",
        "numpy<2.0",
        "packaging",
        "wheel",
    )
    .pip_install("vllm==0.9.1")
    .pip_install(
        "accelerate",
        "openai", 
        "huggingface_hub[hf_transfer]",
        "tokenizers",
        "requests",
        "auto-gptq>=0.7.1",
        "autoawq>=0.2.6", 
        "optimum",
        "scipy",
        "sentencepiece",
        "protobuf",
        "psutil",
        "pynvml",
    )
)

# --- Core chat function ---
def run_chat_logic(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    """Core chat logic that runs the vLLM server and handles chat"""
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
    
    # Read model config
    model_config = get_model_config(model_path)
    
    # Check compatibility
    if not is_model_supported(model_config, model_name):
        print("⚠️ This model may have compatibility issues with vLLM 0.9.1")
        print("💡 Consider using one of these well-supported alternatives:")
        print("   - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("   - microsoft/DialoGPT-medium")
        print("   - hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4")
        print("   - Qwen/Qwen2.5-3B-Instruct")
        print("\n🎯 Attempting to start anyway with conservative settings...")
    
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
    
    # Get configuration
    vllm_config = get_vllm_config(model_name, gpu_type, model_config)
    
    # Build vLLM command
    vllm_command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--model", str(model_path),
        "--served-model-name", model_name,
        "--tensor-parallel-size", str(vllm_config["tensor_parallel_size"]),
        "--max-model-len", str(vllm_config["max_model_len"]),
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--max-num-batched-tokens", str(min(vllm_config["max_num_seqs"] * 256, vllm_config["max_model_len"] * 2)),
        "--disable-log-requests",
        "--tokenizer-mode", "auto",
        "--dtype", vllm_config["dtype"],
        "--enable-prefix-caching",
    ]
    
    if vllm_config.get("trust_remote_code", False):
        vllm_command.append("--trust-remote-code")
    
    if vllm_config["quantization"]:
        vllm_command.extend(["--quantization", vllm_config["quantization"]])
        print(f"🔧 Detected quantization: {vllm_config['quantization']}")
    
    # GPU-specific optimizations
    if gpu_type in ["A100-40GB", "A100-80GB", "H100", "H200", "B200"]:
        vllm_command.append("--enable-chunked-prefill")
    elif gpu_type == "L4":
        vllm_command.extend([
            "--block-size", "16",
            "--swap-space", "4",
        ])
    elif gpu_type == "L40S":
        vllm_command.extend([
            "--block-size", "32",
            "--swap-space", "8",
        ])
    
    print("🚀 Starting vLLM server...")
    print(f"⚙️ Config: max_len={vllm_config['max_model_len']}, tensor_parallel={vllm_config['tensor_parallel_size']}, dtype={vllm_config['dtype']}")
    if vllm_config["quantization"]:
        print(f"🗜️ Quantization: {vllm_config['quantization']}")
    
    # Environment setup
    env = os.environ.copy()
    env["VLLM_USE_MODELSCOPE"] = "False"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    
    # GPU-specific environment settings
    if gpu_type == "L4":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    elif gpu_type in ["H100", "H200", "B200"]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        env["VLLM_USE_TRITON_FLASH_ATTN"] = "1"
    
    vllm_process = subprocess.Popen(
        vllm_command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        env=env
    )
    
    # Wait for startup
    max_retries = 300
    output_lines = []
    
    for i in range(max_retries):
        if vllm_process.poll() is not None:
            try:
                remaining_output = vllm_process.stdout.read()
                if remaining_output:
                    output_lines.extend(remaining_output.split('\n'))
            except:
                pass
            
            print(f"❌ vLLM process terminated with code: {vllm_process.returncode}")
            print(f"📝 Process output (last 40 lines):")
            for line in output_lines[-40:]:
                if line.strip():
                    print(f"   {line.strip()}")
            
            # Error hints
            output_text = '\n'.join(output_lines)
            if "out of memory" in output_text.lower():
                print(f"\n💡 Hint: Try reducing max_model_len or using a larger GPU than {gpu_type}")
            elif "engine core initialization failed" in output_text.lower():
                print(f"\n💡 Hint: Model '{model_name}' may not be fully compatible with vLLM 0.9.1")
                print("   Try these well-supported alternatives:")
                print("   - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                print("   - microsoft/DialoGPT-medium")
                print("   - hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4")
            
            raise RuntimeError(f"vLLM failed to start (exit code: {vllm_process.returncode})")
        
        try:
            line = vllm_process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                if i % 30 == 0 and line.strip():
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
            if i % 60 == 0:
                print(f"⏳ Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("⏰ Startup timeout reached!")
        vllm_process.terminate()
        raise RuntimeError("vLLM server failed to start within timeout period")
    
    # Server ready - continue with API/chat logic
    with modal.forward(port) as tunnel:
        print(f"🌐 Server URL: {tunnel.url}")
        
        # Test API
        import requests
        try:
            test_response = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
            print(f"✅ API test successful: {test_response.status_code}")
            
            models_data = test_response.json()
            available_models = [m.get('id', 'Unknown') for m in models_data.get('data', [])]
            print(f"📋 Available models: {', '.join(available_models)}")
        except Exception as e:
            print(f"⚠️  API test failed: {e}")
        
        # Handle different modes
        if api_only:
            print("\n🌐 API Server ready! Running in API-only mode.")
            print(f"📖 Server URL: {tunnel.url}")
            print("🔌 Available endpoints:")
            print(f"  - Health: {tunnel.url}/health")
            print(f"  - Models: {tunnel.url}/v1/models")
            print(f"  - Chat: {tunnel.url}/v1/chat/completions")
            print(f"  - Completions: {tunnel.url}/v1/completions")
            print("\n⏰ Server running indefinitely. Modal will auto-scale down after inactivity.")
            print("💡 Press Ctrl+C to stop the server manually.")
            
            try:
                while True:
                    time.sleep(300)
                    print("💓 Server heartbeat - still running...")
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                return
            finally:
                vllm_process.terminate()
        
        elif custom_questions is None or len(custom_questions) == 0:
            print("\n🧪 API demo mode - server will stay alive for 5 minutes for testing")
            print(f"📖 Server URL: {tunnel.url}")
            print("🔌 Available endpoints:")
            print(f"  - Health: {tunnel.url}/health")
            print(f"  - Models: {tunnel.url}/v1/models")
            print(f"  - Chat: {tunnel.url}/v1/chat/completions")
            print(f"  - Completions: {tunnel.url}/v1/completions")
            
            print("\n⏰ Keeping server alive for 5 minutes for testing...")
            try:
                time.sleep(300)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
            finally:
                vllm_process.terminate()
            return
        
        else:
            # Chat demo mode 
            questions = custom_questions or [
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
                        headers={"Content-Type": "application/json", "Authorization": "Bearer vllm"},
                        json={
                            "model": model_name,
                            "messages": conversation,
                            "max_tokens": min(300, vllm_config["max_model_len"] // 4),
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
            try:
                time.sleep(300)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
            finally:
                vllm_process.terminate()

# --- GPU-specific functions ---
@app.function(gpu="T4", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_t4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="L4", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_l4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="A10G", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_a10g(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="L40S", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_l40s(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="A100", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_a100_40gb(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="A100:8", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)  
def run_chat_a100_80gb(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="H100", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_h100(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="H200", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_h200(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(gpu="B200", image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_chat_b200(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

# --- Helper function ---
def get_chat_function(model_name: str):
    """Get the appropriate chat function based on model requirements"""
    gpu_type, _ = get_gpu_config(model_name)
    
    function_map = {
        "T4": run_chat_t4,
        "L4": run_chat_l4,
        "A10G": run_chat_a10g,
        "L40S": run_chat_l40s,
        "A100-40GB": run_chat_a100_40gb,
        "A100-80GB": run_chat_a100_80gb, 
        "H100": run_chat_h100,
        "H200": run_chat_h200,
        "B200": run_chat_b200,
    }
    
    return function_map.get(gpu_type, run_chat_l4)

# --- Model management ---
@app.function(image=base_image, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def download_model_remote(model_name: str):
    print(f"📥 Downloading model: {model_name}")
    volume_name, model_base_path, model_path = get_model_info(model_name)
    download_model_to_path(model_name, model_path)
    return f"✅ Model {model_name} downloaded successfully"

# --- Local entrypoints ---
@app.local_entrypoint()
def chat(questions: str = ""):
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🚀 Starting chat session...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    
    custom_questions = None
    if questions:
        custom_questions = [q.strip() for q in questions.split("|") if q.strip()]
        print(f"📝 Using {len(custom_questions)} custom questions")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions, api_only=False)

@app.local_entrypoint()
def serve_api():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🌐 Starting API-only server...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=None, api_only=True)

@app.local_entrypoint()
def serve_demo():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🧪 Starting API demo server (5 minutes)...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=[], api_only=False)

@app.local_entrypoint()
def test_large_model():
    """Test large model with automatic GPU selection"""
    model_name = "unsloth/Meta-Llama-3.1-405B-bnb-4bit"
    print(f"🧪 Testing large model: {model_name}")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! I'm testing a 70B model with automatic GPU selection.",
        "What are the advantages of large language models?",
        "Write Python code for a binary search algorithm.",
        "Explain quantum computing in simple terms.",
        "Thank you for the demonstration!"
    ], api_only=False)

@app.local_entrypoint()
def gpu_specs():
    """Show GPU specifications and recommended models"""
    print("🚀 GPU Specifications & Model Recommendations")
    print("=" * 80)
    
    specs = [
        ("T4", "16GB", "1-2B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        ("L4", "24GB", "3B, 7B-GPTQ", "Qwen/Qwen2.5-3B-Instruct"),
        ("A10G", "24GB", "3-6B, 13B-GPTQ", "01-ai/Yi-1.5-6B-Chat"),
        ("L40S", "48GB", "7B, 13B-GPTQ", "unsloth/mistral-7b-v0.2"),
        ("A100-40GB", "40GB", "7-13B, 34B-GPTQ", "unsloth/llama-2-13b"),
        ("A100-80GB", "80GB", "13-34B, 70B-GPTQ", "unsloth/gemma-3-27b-it"),
        ("H100", "80GB", "34-70B", "unsloth/Llama-4-Maverick-17B-128E-Instruct"),
        ("H200", "141GB", "70B+", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
        ("B200", "192GB", "405B", "unsloth/Meta-Llama-3.1-405B-bnb-4bit"),
    ]
    
    for gpu, memory, models, example in specs:
        print(f"\n🔧 {gpu:8} | {memory:7} | {models:15} | {example}")
    
    print(f"\n💡 Usage Examples:")
    print(f"  Small:   MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0' modal run script.py::serve_api")
    print(f"  Medium:  MODEL_NAME='Qwen/Qwen2.5-3B-Instruct' modal run script.py::serve_api")
    print(f"  Large:   MODEL_NAME='hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4' modal run script.py::serve_api")

@app.local_entrypoint()
def download(model_name: str = DEFAULT_MODEL):
    print(f"📥 Downloading model: {model_name}")
    result = download_model_remote.remote(model_name)
    print(result)

@app.local_entrypoint()
def info():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, gpu_util = get_gpu_config(current_model)
    vllm_config = get_vllm_config(current_model, gpu_type)
    
    print(f"🚀 vLLM v0.9.1 Configuration:")
    print(f"  Model: {current_model}")
    print(f"  GPU: {gpu_type} ({gpu_util*100}% memory)")
    print(f"  Max length: {vllm_config['max_model_len']}")
    print(f"  Tensor parallel: {vllm_config['tensor_parallel_size']}")
    print(f"  Dtype: {vllm_config['dtype']}")
    if vllm_config['quantization']:
        print(f"  Quantization: {vllm_config['quantization']}")
    
    print(f"\n📋 Available Commands:")
    print(f"  serve_api       - Run API server indefinitely")
    print(f"  serve_demo      - Run API server for 5 minutes")
    print(f"  chat            - Run chat demo + 5 min API")
    print(f"  test_large_model - Test 70B model with auto GPU selection")
    print(f"  gpu_specs       - Show all GPU specifications")
    print(f"  download        - Download model only")
    print(f"  info            - Show this configuration")
