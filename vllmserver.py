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
            
            print(f"📖 Model config: type={model_type}, max_pos_emb={max_position_embeddings}, max_len={model_max_length}")
            
            return {
                "max_position_embeddings": max_position_embeddings,
                "model_max_length": model_max_length,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "model_type": model_type
            }
        except Exception as e:
            print(f"⚠️ Could not read model config: {e}")
            return None
    return None

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
    
    # Large models (7B+ parameters) - check for quantized versions
    if any(size in model_name_lower for size in ["7b", "8b", "13b", "70b"]):
        # Quantized models need less memory
        if any(quant in model_name_lower for quant in ["gptq", "int4", "int8", "awq", "gguf"]):
            if "70b" in model_name_lower:
                return "A100", 0.85  # 70B quantized can fit on A100
            elif any(size in model_name_lower for size in ["13b"]):
                return "A10G", 0.8  # 13B quantized needs A10G
            elif any(size in model_name_lower for size in ["7b", "8b"]):
                return "L4", 0.75   # 7B-8B quantized can fit on L4 with conservative memory
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
        # 3B can fit on L4, 4B+ need A10G
        if any(size in model_name_lower for size in ["3b"]):
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
        "quantization": None,  # Will be set if quantized model detected
    }
    
    model_lower = model_name.lower()
    
    # Use actual model config if available
    if model_config:
        max_pos_emb = model_config.get("max_position_embeddings")
        model_max_len = model_config.get("model_max_length")
        
        # Determine the actual maximum length we can use
        if max_pos_emb:
            # Use 90% of max_position_embeddings for safety
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
        config["quantization"] = "gptq"  # INT4 usually means GPTQ
    elif "int8" in model_lower:
        config["quantization"] = "gptq"
    elif "gguf" in model_lower:
        config["quantization"] = None  # GGUF handled differently in vLLM 0.9+
    
    # Adjust based on model size
    if "70b" in model_lower:
        if not model_config:  # Only adjust if we don't have actual config
            config["max_model_len"] = min(config["max_model_len"], 4096 if config["quantization"] else 2048)
        config["max_num_seqs"] = 8 if config["quantization"] else 4
        config["tensor_parallel_size"] = 2 if gpu_type in ["H100", "A100"] else 1
    elif any(size in model_lower for size in ["7b", "8b", "13b"]):
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 16 if config["quantization"] else 8
    elif any(size in model_lower for size in ["3b", "4b", "6b"]):
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 6144)
        config["max_num_seqs"] = 12
    elif any(size in model_lower for size in ["1b", "2b", "tinyllama"]):
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 8192)
        config["max_num_seqs"] = 16
    elif "dialog" in model_lower or "chat" in model_lower:
        # Dialog models often have smaller context windows
        if not model_config:
            config["max_model_len"] = min(config["max_model_len"], 1024)
        config["max_num_seqs"] = 8
    
    # Adjust based on GPU capabilities
    if gpu_type == "T4":
        config["max_model_len"] = min(config["max_model_len"], 2048)
        config["max_num_seqs"] = min(config["max_num_seqs"], 8)
        config["dtype"] = "half"  # Force float16 for T4 compatibility
    elif gpu_type == "L4":
        # L4 has 24GB VRAM, similar to A10G but newer architecture
        config["max_model_len"] = min(config["max_model_len"], 4096)
        config["max_num_seqs"] = min(config["max_num_seqs"], 12)
        config["dtype"] = "auto"  # L4 supports modern dtypes
    elif gpu_type in ["A100", "H100"]:
        config["dtype"] = "auto"  # These support bfloat16
    else:  # A10G and others
        config["dtype"] = "auto"  # vLLM 0.9+ handles dtype selection better
    
    # Ensure minimum viable values
    config["max_model_len"] = max(config["max_model_len"], 256)
    
    return config

# --- Container image with clean dependency resolution ---
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git")
    .pip_install(
        # Install PyTorch first with exact version for stability
        "torch==2.5.1",
        # Basic numerical dependencies
        "numpy<2.0",
        "packaging",
        "wheel",
    )
    .pip_install(
        # Install vLLM and let it resolve its own transformers version
        "vllm==0.9.1",
    )
    .pip_install(
        # Install remaining packages and let them auto-resolve versions
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
    
    # Read model config after download
    model_config = get_model_config(model_path)
    
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
    
    # Get dynamic configuration with model config
    vllm_config = get_vllm_config(model_name, gpu_type, model_config)
    
    # Build vLLM command with GPU-specific settings
    vllm_command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--model", str(model_path),
        "--served-model-name", model_name,
        "--tensor-parallel-size", str(vllm_config["tensor_parallel_size"]),
        "--trust-remote-code",
        "--max-model-len", str(vllm_config["max_model_len"]),
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--max-num-batched-tokens", str(min(vllm_config["max_num_seqs"] * 256, vllm_config["max_model_len"] * 2)),
        "--disable-log-requests",
        "--tokenizer-mode", "auto",
        "--dtype", vllm_config["dtype"],
        "--enable-prefix-caching",
    ]
    
    # Add quantization if detected
    if vllm_config["quantization"]:
        vllm_command.extend(["--quantization", vllm_config["quantization"]])
        print(f"🔧 Detected quantization: {vllm_config['quantization']}")
    
    # GPU-specific optimizations
    if gpu_type in ["A100", "H100"]:
        vllm_command.append("--enable-chunked-prefill")
    elif gpu_type == "L4":
        # L4 specific optimizations
        vllm_command.extend([
            "--block-size", "16",  # Smaller block size for better memory efficiency
            "--swap-space", "4",   # Enable some CPU swap for larger models
        ])
    
    print("🚀 Starting vLLM server...")
    print(f"⚙️ Config: max_len={vllm_config['max_model_len']}, batched_tokens={min(vllm_config['max_num_seqs'] * 256, vllm_config['max_model_len'] * 2)}, dtype={vllm_config['dtype']}")
    if vllm_config["quantization"]:
        print(f"🗜️ Quantization: {vllm_config['quantization']}")
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env["VLLM_USE_MODELSCOPE"] = "False"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Allow overriding max model len if needed (though we shouldn't need this now)
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    
    # L4-specific environment optimizations
    if gpu_type == "L4":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    vllm_process = subprocess.Popen(
        vllm_command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        env=env
    )
    
    # Wait for startup with monitoring
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
            if "max_model_len" in output_text and "greater than" in output_text:
                print("\n💡 Hint: Model context length issue - this should now be fixed with config reading")
            elif "out of memory" in output_text.lower():
                print(f"\n💡 Hint: Try reducing max_model_len or using a larger GPU than {gpu_type}")
                if gpu_type == "L4":
                    print("💡 For L4: Consider using INT4/GPTQ quantized models for better memory efficiency")
            elif "quantization" in output_text.lower():
                print("\n💡 Hint: Check if the quantization format is supported")
            elif "import" in output_text.lower() and "error" in output_text.lower():
                print("\n💡 Hint: Package dependency issue")
            
            raise RuntimeError(f"vLLM failed to start (exit code: {vllm_process.returncode})")
        
        # Read any new output
        try:
            line = vllm_process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                # Show output every 30 iterations
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
            
            # Show available model
            models_data = test_response.json()
            available_models = [m.get('id', 'Unknown') for m in models_data.get('data', [])]
            print(f"📋 Available models: {', '.join(available_models)}")
        except Exception as e:
            print(f"⚠️  API test failed: {e}")
        
        # Handle different modes
        if api_only:
            # API-only mode - no chat demo, run indefinitely
            print("\n🌐 API Server ready! Running in API-only mode.")
            print(f"📖 Use the following URL for API calls: {tunnel.url}")
            print("🔌 Available endpoints:")
            print(f"  - Health: {tunnel.url}/health")
            print(f"  - Models: {tunnel.url}/v1/models") 
            print(f"  - Chat: {tunnel.url}/v1/chat/completions")
            print(f"  - Completions: {tunnel.url}/v1/completions")
            print("\n⏰ Server running indefinitely. Modal will auto-scale down after inactivity.")
            print("💡 Press Ctrl+C to stop the server manually.")
            
            try:
                # Keep server alive indefinitely
                while True:
                    time.sleep(300)  # 5 minute heartbeat
                    print("💓 Server heartbeat - still running...")
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                return
            finally:
                vllm_process.terminate()
        
        elif custom_questions is None or len(custom_questions) == 0:
            # Empty questions = short API demo mode
            print("\n🧪 API demo mode - server will stay alive for 5 minutes for testing")
            print(f"📖 Use the following URL for API calls: {tunnel.url}")
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
            elif "dialog" in model_name.lower() or "chat" in model_name.lower():
                questions = [
                    "Hello! How are you today?",
                    "What's your favorite hobby?",
                    "Can you tell me a fun fact?",
                    "Thanks for chatting!"
                ]
            else:
                questions = [
                    "Hello! Please introduce yourself briefly.",
                    "What can you help me with today?",
                    "Thank you for the demo!"
                ]
            
            # Use custom questions if provided
            if custom_questions:
                questions = custom_questions
        
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

# --- GPU-specific chat functions ---
@app.function(
    gpu="T4",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_t4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    """Run chat session on T4 GPU"""
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="L4",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_l4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    """Run chat session on L4 GPU"""
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A10G",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_a10g(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    """Run chat session on A10G GPU"""
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A100",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_a100(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    """Run chat session on A100 GPU"""
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="H100",
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_chat_h100(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    """Run chat session on H100 GPU"""
    return run_chat_logic(model_name, custom_questions, api_only)

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
    elif gpu_type == "L4":
        return run_chat_l4
    elif gpu_type == "A10G":
        return run_chat_a10g
    elif gpu_type == "A100":
        return run_chat_a100
    elif gpu_type == "H100":
        return run_chat_h100
    else:
        return run_chat_l4  # Default to L4 as it's cost-effective

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
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions, api_only=False)

@app.local_entrypoint()
def serve_api():
    """Start API server without chat demo - keeps running until stopped"""
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🌐 Starting API-only server...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    print(f"⏰ Server will run indefinitely until manually stopped")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=None, api_only=True)

@app.local_entrypoint()
def serve_demo():
    """Start API server for 5 minutes - good for testing"""
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🧪 Starting API demo server (5 minutes)...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=[], api_only=False)

@app.local_entrypoint()
def test_llama31():
    """Test Llama 3.1 8B GPTQ with vLLM 0.9.1"""
    model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    print(f"🧪 Testing {model_name} with vLLM 0.9.1")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! I'm testing Llama 3.1 with vLLM 0.9.1. Please introduce yourself.",
        "What new features does Llama 3.1 bring?",
        "Write a Python function to calculate factorial using recursion.",
        "Explain machine learning in simple terms.",
        "Thanks for the comprehensive test!"
    ], api_only=False)

@app.local_entrypoint()
def test_l4():
    """Test L4 with a quantized model optimized for its memory"""
    model_name = "microsoft/DialoGPT-medium"  # Good test model for L4
    print(f"🧪 Testing L4 GPU with {model_name}")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! I'm testing L4 GPU capabilities.",
        "What's the weather like?",
        "Tell me a short story about AI.",
        "Thanks for the L4 test!"
    ], api_only=False)

@app.local_entrypoint()
def download(model_name: str = DEFAULT_MODEL):
    """Download a model without starting server"""
    print(f"📥 Downloading model: {model_name}")
    result = download_model_remote.remote(model_name)
    print(result)

@app.local_entrypoint()
def curl_examples():
    """Show curl examples for testing the API"""
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    
    print(f"🌐 Curl Examples for Testing vLLM API")
    print(f"📖 Model: {current_model}")
    print("=" * 60)
    
    print("\n1. 🏥 Health Check:")
    print("curl -f https://your-server-url.modal.run/health")
    
    print("\n2. 📋 List Models:")
    print("curl https://your-server-url.modal.run/v1/models")
    
    print("\n3. 💬 Chat Completion:")
    print(f"""curl https://your-server-url.modal.run/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer vllm" \\
  -d '{{
    "model": "{current_model}",
    "messages": [
      {{"role": "user", "content": "Hello! Can you introduce yourself?"}}
    ],
    "max_tokens": 200,
    "temperature": 0.8
  }}'""")
    
    print("\n4. ✍️  Text Completion:")
    print(f"""curl https://your-server-url.modal.run/v1/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer vllm" \\
  -d '{{
    "model": "{current_model}",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }}'""")
    
    print("\n5. 🌊 Streaming Chat:")
    print(f"""curl https://your-server-url.modal.run/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer vllm" \\
  -d '{{
    "model": "{current_model}",
    "messages": [
      {{"role": "user", "content": "Write a Python function for fibonacci"}}
    ],
    "max_tokens": 300,
    "stream": true
  }}'""")
    
    print("\n💡 Tips:")
    print("- Replace 'your-server-url.modal.run' with the actual URL from the server output")
    print("- Use 'Bearer vllm' as the authorization header")
    print(f"- Model name should be: {current_model}")
    
    print(f"\n🚀 Server Commands:")
    print(f"  API-only (indefinite):  MODEL_NAME='{current_model}' modal run script.py::serve_api")
    print(f"  API demo (5 min):       MODEL_NAME='{current_model}' modal run script.py::serve_demo")
    print(f"  Chat demo:              MODEL_NAME='{current_model}' modal run script.py::chat")

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
    print(f"  Dtype: {vllm_config['dtype']}")
    if vllm_config['quantization']:
        print(f"  Quantization: {vllm_config['quantization']}")
    
    print(f"\n🎯 GPU Capabilities:")
    print(f"  T4:  16GB VRAM  - Small models (1-2B), basic quantized")
    print(f"  L4:  24GB VRAM  - Medium models (3B), 7B quantized")
    print(f"  A10G: 24GB VRAM - Medium models (3-6B), 7B quantized")
    print(f"  A100: 40-80GB   - Large models (7-13B), 70B quantized")
    print(f"  H100: 80GB      - Largest models (70B+)")
    
    print(f"\n📋 Available Commands:")
    print(f"  serve_api    - Run API server indefinitely")
    print(f"  serve_demo   - Run API server for 5 minutes") 
    print(f"  chat         - Run chat demo + 5 min API")
    print(f"  test_llama31 - Test Llama 3.1 model")
    print(f"  test_l4      - Test L4 GPU specifically")
    print(f"  download     - Download model only")
    print(f"  curl_examples - Show API usage examples")
    print(f"  info         - Show this configuration")
    
    print(f"\n💡 L4 Recommended Models:")
    print(f"  - microsoft/DialoGPT-medium")
    print(f"  - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"  - stabilityai/stablelm-3b-4e1t")
    print(f"  - Qwen/Qwen2-3B-Instruct")
    
    print(f"\n🔧 Note: Actual max_model_len will be adjusted based on model config.json after download")
