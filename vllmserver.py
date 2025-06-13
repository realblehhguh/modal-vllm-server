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

# --- Create a default volume for the app ---
default_volume = modal.Volume.from_name("vllm-models-storage", create_if_missing=True)

# --- Helper functions ---
def get_model_path_from_name(model_name: str):
    """Get a safe path for the model inside the volume"""
    # Create a safe directory name from model name
    safe_name = model_name.replace("/", "--").replace("_", "-").replace(".", "-").lower()
    model_base_path = Path("/models")
    model_path = model_base_path / safe_name
    return model_path

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

def check_model_sharding(model_path: Path):
    """Check if model is sharded and get shard info"""
    index_path = model_path / "pytorch_model.bin.index.json"
    safetensors_index_path = model_path / "model.safetensors.index.json"
    
    # Check for sharded PyTorch models
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            shard_files = set(weight_map.values())
            print(f"📊 Found PyTorch sharded model with {len(shard_files)} shards:")
            for i, shard in enumerate(sorted(shard_files)[:5], 1):  # Show first 5
                shard_path = model_path / shard
                if shard_path.exists():
                    size_mb = shard_path.stat().st_size / 1e6
                    print(f"   ✅ {shard} ({size_mb:.1f} MB)")
                else:
                    print(f"   ❌ {shard} (missing)")
            if len(shard_files) > 5:
                print(f"   ... and {len(shard_files) - 5} more shards")
            return True, len(shard_files), list(shard_files)
        except Exception as e:
            print(f"⚠️ Could not read PyTorch shard index: {e}")
    
    # Check for sharded SafeTensors models
    elif safetensors_index_path.exists():
        try:
            with open(safetensors_index_path, 'r') as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            shard_files = set(weight_map.values())
            print(f"📊 Found SafeTensors sharded model with {len(shard_files)} shards:")
            for i, shard in enumerate(sorted(shard_files)[:5], 1):  # Show first 5
                shard_path = model_path / shard
                if shard_path.exists():
                    size_mb = shard_path.stat().st_size / 1e6
                    print(f"   ✅ {shard} ({size_mb:.1f} MB)")
                else:
                    print(f"   ❌ {shard} (missing)")
            if len(shard_files) > 5:
                print(f"   ... and {len(shard_files) - 5} more shards")
            return True, len(shard_files), list(shard_files)
        except Exception as e:
            print(f"⚠️ Could not read SafeTensors shard index: {e}")
    
    # Check for simple sharded files (no index)
    pytorch_shards = list(model_path.glob("pytorch_model-*.bin"))
    safetensors_shards = list(model_path.glob("model-*.safetensors"))
    
    if pytorch_shards:
        print(f"📊 Found {len(pytorch_shards)} PyTorch shard files (no index):")
        for shard in sorted(pytorch_shards)[:5]:
            size_mb = shard.stat().st_size / 1e6
            print(f"   ✅ {shard.name} ({size_mb:.1f} MB)")
        if len(pytorch_shards) > 5:
            print(f"   ... and {len(pytorch_shards) - 5} more shards")
        return True, len(pytorch_shards), [s.name for s in pytorch_shards]
    
    elif safetensors_shards:
        print(f"📊 Found {len(safetensors_shards)} SafeTensors shard files (no index):")
        for shard in sorted(safetensors_shards)[:5]:
            size_mb = shard.stat().st_size / 1e6
            print(f"   ✅ {shard.name} ({size_mb:.1f} MB)")
        if len(safetensors_shards) > 5:
            print(f"   ... and {len(safetensors_shards) - 5} more shards")
        return True, len(safetensors_shards), [s.name for s in safetensors_shards]
    
    else:
        # Check for single model files
        pytorch_single = model_path / "pytorch_model.bin"
        safetensors_single = model_path / "model.safetensors"
        
        if pytorch_single.exists():
            size_mb = pytorch_single.stat().st_size / 1e6
            print(f"📊 Found single PyTorch model file: pytorch_model.bin ({size_mb:.1f} MB)")
            return False, 1, ["pytorch_model.bin"]
        elif safetensors_single.exists():
            size_mb = safetensors_single.stat().st_size / 1e6
            print(f"📊 Found single SafeTensors model file: model.safetensors ({size_mb:.1f} MB)")
            return False, 1, ["model.safetensors"]
        else:
            print("❌ No model weights found!")
            return False, 0, []

def download_model_to_path(model_name: str, model_path: Path):
    """Download model to specific path with integrity checking for sharded models"""
    import os
    import shutil
    from huggingface_hub import snapshot_download
    
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Check if model already exists and is complete
    if model_path.exists():
        print(f"📁 Model directory exists at {model_path}, checking integrity...")
        
        # Check for essential files
        essential_files = ["config.json"]
        missing_essential = []
        for file in essential_files:
            file_path = model_path / file
            if not file_path.exists() or file_path.stat().st_size == 0:
                missing_essential.append(file)
        
        if missing_essential:
            print(f"❌ Found missing essential files: {missing_essential}")
            print(f"❌ Re-downloading...")
            print("🗑️  Removing corrupted model directory...")
            shutil.rmtree(model_path)
        else:
            # Check model weights integrity
            is_sharded, shard_count, shard_files = check_model_sharding(model_path)
            
            if shard_count == 0:
                print(f"❌ No model weights found, re-downloading...")
                shutil.rmtree(model_path)
            else:
                print("✅ Model integrity check passed - using cached model")
                if is_sharded:
                    print(f"🔗 Sharded model with {shard_count} shards detected")
                return

    if not model_path.exists():
        print(f"📥 Downloading {model_name} to {model_path}...")
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download to path
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                token=os.environ.get("HF_TOKEN"),
                resume_download=True,  # Resume interrupted downloads
                local_files_only=False,
            )
            
            print("✅ Download complete!")
            
            # Check what we downloaded
            is_sharded, shard_count, shard_files = check_model_sharding(model_path)
            if is_sharded:
                print(f"🔗 Downloaded sharded model with {shard_count} parts")
            else:
                print("📦 Downloaded single-file model")
            
            print("💾 Model saved to persistent volume!")
            
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
        if any(quant in model_name_lower for quant in ["gptq", "int4", "int8", "awq", "bnb"]):
            if "405b" in model_name_lower:
                return "B200", 0.90  # 405B quantized needs B200
            elif any(size in model_name_lower for size in ["70b", "72b"]):
                return "H200", 0.85  # 70B quantized can fit on H200
        else:
            # Full precision massive models
            if "405b" in model_name_lower:
                return "B200", 0.95  # 405B full precision needs B200
            elif any(size in model_name_lower for size in ["70b", "72b"]):
                return "A100-80GB", 0.90  # 70B full precision needs A100-80GB
    
    # Large models (13B-34B parameters)
    elif any(size in model_name_lower for size in ["13b", "14b", "17b", "27b", "34b"]):
        if any(quant in model_name_lower for quant in ["gptq", "int4", "int8", "awq"]):
            if any(size in model_name_lower for size in ["27b", "34b"]):
                return "A100-80GB", 0.85  # 27B+ quantized needs A100-80GB
            elif any(size in model_name_lower for size in ["17b"]):
                return "H100", 0.80  # 17B quantized fits on H100
            elif any(size in model_name_lower for size in ["13b", "14b"]):
                return "A100-40GB", 0.80  # 13B quantized fits on A100-40GB
        else:
            # Full precision large models
            if any(size in model_name_lower for size in ["27b", "34b"]):
                return "A100-80GB", 0.85  # 27B+ full precision needs A100-80GB
            elif any(size in model_name_lower for size in ["17b"]):
                return "H100", 0.85  # 17B full precision needs H100
            elif any(size in model_name_lower for size in ["13b", "14b"]):
                return "A100-40GB", 0.80  # 13B full precision needs A100-40GB
    
    # Medium-large models (7B-12B parameters)
    elif any(size in model_name_lower for size in ["7b", "8b", "9b", "12b"]):
        if any(quant in model_name_lower for quant in ["gptq", "int4", "int8", "awq"]):
            return "L40S", 0.75   # 7B-12B quantized fits on L40S
        else:
            # Full precision medium-large models
            return "L40S", 0.80  # 7B-12B full precision works well on L40S
    
    # Medium models (3-6B parameters)  
    elif any(size in model_name_lower for size in ["3b", "4b", "6b"]):
        if "stablelm" in model_name_lower or "stablecode" in model_name_lower:
            return "A10G", 0.8  # StableLM models need more resources
        elif any(size in model_name_lower for size in ["6b"]):
            return "A10G", 0.75  # 6B models work well on A10G
        elif any(size in model_name_lower for size in ["3b", "4b"]):
            return "L4", 0.75
        else:
            return "A10G", 0.8
    
    # Small models (1-2B parameters)
    elif any(size in model_name_lower for size in ["1b", "2b", "mini", "small", "tinyllama"]):
        return "T4", 0.8
    
    # Chat/dialog models (usually medium sized)
    elif any(term in model_name_lower for term in ["dialog", "chat", "gpt2", "dialo"]):
        return "L4", 0.75  # Most chat models are small-medium
    
    # Default for unknown sizes
    else:
        return "L4", 0.75  # Conservative default

# --- Dynamic vLLM configuration based on model ---
def get_vllm_config(model_name: str, gpu_type: str, model_config: dict = None, is_sharded: bool = False):
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
    elif "int4" in model_lower or "bnb" in model_lower:
        config["quantization"] = "gptq"
    elif "int8" in model_lower:
        config["quantization"] = "gptq"
    elif "gguf" in model_lower:
        config["quantization"] = None
    
    # Sharded models may need special handling
    if is_sharded:
        print("🔗 Detected sharded model - adjusting configuration for multi-file loading")
        # Sharded models typically benefit from more conservative memory settings
        config["max_num_seqs"] = max(2, config["max_num_seqs"] // 2)
    
    # Model size based adjustments
    if any(size in model_lower for size in ["405b"]):
        config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 16 if config["quantization"] else 8
        config["tensor_parallel_size"] = 8 if gpu_type == "B200" else 4
    elif any(size in model_lower for size in ["70b", "72b"]):
        config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 12 if config["quantization"] else 6
        config["tensor_parallel_size"] = 4 if gpu_type in ["H100", "H200", "A100-80GB", "B200"] else 2
    elif any(size in model_lower for size in ["27b", "34b"]):
        config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 10 if config["quantization"] else 6
        config["tensor_parallel_size"] = 2 if gpu_type in ["A100-40GB", "A100-80GB", "H100", "H200", "L40S"] else 1
    elif any(size in model_lower for size in ["17b"]):
        config["max_model_len"] = min(config["max_model_len"], 8192 if config["quantization"] else 4096)
        config["max_num_seqs"] = 8 if config["quantization"] else 6
        config["tensor_parallel_size"] = 2 if gpu_type in ["H100", "H200", "A100-80GB"] else 1
    elif any(size in model_lower for size in ["7b", "8b", "9b", "12b", "13b", "14b"]):
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
    elif gpu_type == "H100":
        config["max_model_len"] = min(config["max_model_len"], 16384)
        config["max_num_seqs"] = min(config["max_num_seqs"], 48)
        config["dtype"] = "auto"
    elif gpu_type == "H200":
        config["max_model_len"] = min(config["max_model_len"], 20480)
        config["max_num_seqs"] = min(config["max_num_seqs"], 56)
        config["dtype"] = "auto"
    elif gpu_type == "B200":
        config["max_model_len"] = min(config["max_model_len"], 32768)
        config["max_num_seqs"] = min(config["max_num_seqs"], 64)
        config["dtype"] = "auto"
    
    # Ensure minimum viable values
    config["max_model_len"] = max(config["max_model_len"], 256)
    
    return config

# --- Container image ---
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
    model_path = get_model_path_from_name(model_name)
    
    port = 8000
    
    print(f"🤖 Using model: {model_name}")
    print(f"🔧 GPU: {gpu_type}, Memory utilization: {gpu_memory_util}")
    print(f"📁 Model path: {model_path}")
    print(f"📦 Using persistent volume for model storage")
    
    # Download model if it doesn't exist
    if not model_path.exists():
        print(f"📥 Model not found in volume, downloading...")
        download_model_to_path(model_name, model_path)
    else:
        print(f"✅ Model found in volume: {model_path}")
    
    if not model_path.exists():
        raise RuntimeError(f"Model path {model_path} does not exist after download")
    
    # Check if model is sharded
    is_sharded, shard_count, shard_files = check_model_sharding(model_path)
    
    # Read model config
    model_config = get_model_config(model_path)
    
    # Check compatibility
    if not is_model_supported(model_config, model_name):
        print("⚠️ This model may have compatibility issues with vLLM 0.9.1")
        print("💡 Consider using one of these well-supported alternatives:")
        print("   - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("   - Qwen/Qwen2.5-3B-Instruct")
        print("   - 01-ai/Yi-1.5-6B-Chat")
        print("   - unsloth/mistral-7b-v0.2")
        print("\n🎯 Attempting to start anyway with conservative settings...")
    
    # Show model files summary
    files = list(model_path.glob("*"))
    print(f"📁 Model directory contains {len(files)} files")
    
    # Calculate total size
    total_size = 0
    for file in files:
        if file.is_file():
            total_size += file.stat().st_size
    
    print(f"💾 Total model size: {total_size / 1e9:.2f} GB")
    if is_sharded:
        print(f"🔗 Model is sharded across {shard_count} files")
    
    # Special handling for large sharded models that get stuck
    if is_sharded and shard_count >= 4 and total_size > 15e9:  # >15GB
        print(f"⚠️ Large sharded model detected - applying compatibility fixes")
        # Force single GPU loading for large sharded models
        gpu_memory_util = min(gpu_memory_util, 0.75)  # More conservative memory
        print(f"🔧 Reduced GPU memory utilization to {gpu_memory_util} for stability")
    
    # Get configuration
    vllm_config = get_vllm_config(model_name, gpu_type, model_config, is_sharded)
    
    # Additional fixes for large sharded models
    if is_sharded and shard_count >= 4:
        # Force single GPU for problematic sharded models
        vllm_config["tensor_parallel_size"] = 1
        # More conservative sequence settings
        vllm_config["max_num_seqs"] = max(2, vllm_config["max_num_seqs"] // 4)
        print(f"🔧 Applied large sharded model fixes: TP=1, max_seqs={vllm_config['max_num_seqs']}")
    
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
        "--max-num-batched-tokens", str(min(vllm_config["max_num_seqs"] * 128, vllm_config["max_model_len"])),  # Reduced batch size
        "--disable-log-requests",
        "--tokenizer-mode", "auto",
        "--dtype", vllm_config["dtype"],
    ]
    
    # Conditional flags for better compatibility
    if not (is_sharded and shard_count >= 4):
        vllm_command.append("--enable-prefix-caching")
    else:
        print("🔧 Disabled prefix caching for large sharded model compatibility")
    
    if vllm_config.get("trust_remote_code", False):
        vllm_command.append("--trust-remote-code")
    
    if vllm_config["quantization"]:
        vllm_command.extend(["--quantization", vllm_config["quantization"]])
        print(f"🔧 Detected quantization: {vllm_config['quantization']}")
    
    # Special handling for large sharded models
    if is_sharded and shard_count >= 4:
        print(f"🔗 Large sharded model detected ({shard_count} shards)")
        # Force specific load format
        vllm_command.extend(["--load-format", "safetensors"])
        # Disable problematic optimizations
        vllm_command.extend(["--disable-custom-all-reduce"])
        # Use dummy scheduler for compatibility
        vllm_command.extend(["--max-num-seqs", str(vllm_config["max_num_seqs"])])
    else:
        # Standard optimizations for smaller models
        if gpu_type in ["A100-40GB", "A100-80GB", "H100", "H200", "B200"]:
            vllm_command.append("--enable-chunked-prefill")
        elif gpu_type == "L4":
            vllm_command.extend([
                "--block-size", "16",
                "--swap-space", "4",
            ])
        elif gpu_type == "L40S":
            vllm_command.extend([
                "--block-size", "16",  # Reduced from 32 for sharded models
                "--swap-space", "4",   # Reduced swap space
            ])
    
    print("🚀 Starting vLLM server...")
    print(f"⚙️ Config: max_len={vllm_config['max_model_len']}, tensor_parallel={vllm_config['tensor_parallel_size']}, dtype={vllm_config['dtype']}")
    if vllm_config["quantization"]:
        print(f"🗜️ Quantization: {vllm_config['quantization']}")
    if is_sharded:
        print(f"🔗 Sharded model loading enabled with compatibility fixes")
    
    # Environment setup
    env = os.environ.copy()
    env["VLLM_USE_MODELSCOPE"] = "False"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    
    # Sharded model environment optimizations
    if is_sharded:
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
        env["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer conflicts
        # Force single process for large sharded models
        if shard_count >= 4:
            env["VLLM_USE_RAY_COMPILED_DAG"] = "0"
            env["VLLM_USE_RAY_SPMD_WORKER"] = "0"
            env["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "1"
            print("🔧 Applied single-process fixes for large sharded model")
    
    # GPU-specific environment settings  
    if gpu_type == "L4":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    elif gpu_type == "L40S":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # Reduced for sharded models
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    elif gpu_type in ["H100", "H200", "B200"]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        env["VLLM_USE_TRITON_FLASH_ATTN"] = "1"
    
    # Add timeout handling for stuck processes
    print(f"🔧 Command: {' '.join(vllm_command)}")
    
    vllm_process = subprocess.Popen(
        vllm_command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        env=env
    )
    
    # Wait for startup (longer timeout for sharded models)
    max_retries = 600 if is_sharded and shard_count >= 4 else 300  # Extended to 20 minutes for large sharded
    output_lines = []
    stuck_counter = 0
    last_output_time = time.time()
    
    if is_sharded:
        print(f"⏳ Large sharded model loading may take 10-20 minutes...")
        print(f"⏰ Will wait up to {max_retries//30} minutes for loading to complete")
    
    for i in range(max_retries):
        if vllm_process.poll() is not None:
            try:
                remaining_output = vllm_process.stdout.read()
                if remaining_output:
                    output_lines.extend(remaining_output.split('\n'))
            except:
                pass
            
            print(f"❌ vLLM process terminated with code: {vllm_process.returncode}")
            print(f"📝 Process output (last 50 lines):")
            for line in output_lines[-50:]:
                if line.strip():
                    print(f"   {line.strip()}")
            
            # Error hints
            output_text = '\n'.join(output_lines)
            if "out of memory" in output_text.lower():
                print(f"\n💡 Hint: GPU memory exhausted. Try a smaller model or reduce max_model_len")
                if is_sharded:
                    print(f"💡 Large sharded models need substantial memory. Consider using A100-80GB or H100.")
            elif "engine core initialization failed" in output_text.lower():
                print(f"\n💡 Hint: Model '{model_name}' may not be fully compatible with vLLM 0.9.1")
                print("   Try these well-supported alternatives:")
                print("   - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                print("   - Qwen/Qwen2.5-3B-Instruct")
                print("   - 01-ai/Yi-1.5-6B-Chat")
            elif is_sharded and ("load" in output_text.lower() or "cuda" in output_text.lower()):
                print(f"\n💡 Sharded model CUDA initialization issue.")
                print(f"   Try: MODEL_NAME='Qwen/Qwen2.5-3B-Instruct' modal run script.py::serve_api")
                print(f"   Or use a smaller non-sharded model first")
            
            raise RuntimeError(f"vLLM failed to start (exit code: {vllm_process.returncode})")
        
        try:
            line = vllm_process.stdout.readline()
            if line and line.strip():
                output_lines.append(line.strip())
                last_output_time = time.time()
                stuck_counter = 0
                
                # Show important progress lines
                if any(keyword in line.lower() for keyword in ["loading", "shard", "cuda", "memory", "model", "initialized"]):
                    print(f"   🔗 {line.strip()}")
                elif i % 30 == 0:  # Show periodic updates
                    print(f"   {line.strip()}")
            else:
                # Check if we're stuck (no output for too long)
                if time.time() - last_output_time > 300:  # 5 minutes without output
                    stuck_counter += 1
                    if stuck_counter >= 3:
                        print(f"⚠️ Process appears stuck (no output for {(time.time() - last_output_time):.0f}s)")
                        print(f"💡 This model may not be compatible. Try a smaller model:")
                        print(f"   MODEL_NAME='Qwen/Qwen2.5-3B-Instruct' modal run script.py::serve_api")
                        vllm_process.terminate()
                        raise RuntimeError("Process stuck during model loading")
        except:
            pass
        
        # Test if server is ready
        try:
            result = subprocess.run(
                ["curl", "-f", f"http://localhost:{port}/health"],
                check=True, capture_output=True, timeout=5
            )
            print("✅ vLLM server is ready!")
            if is_sharded:
                print(f"🔗 Sharded model loaded successfully!")
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if i % 60 == 0:
                elapsed_min = (i * 2) // 60
                print(f"⏳ Waiting for server... ({elapsed_min} min elapsed, max {max_retries//30} min)")
                if is_sharded and i > 0:
                    print(f"   🔗 Large sharded models can take 10-20 minutes to load...")
            time.sleep(2)
    else:
        print("⏰ Startup timeout reached!")
        print(f"💡 The Yi-1.5-9B model may be too large/complex for vLLM 0.9.1")
        print(f"💡 Try a smaller, non-sharded model:")
        print(f"   MODEL_NAME='Qwen/Qwen2.5-3B-Instruct' modal run script.py::serve_api")
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
            if is_sharded:
                print(f"🔗 Serving sharded model with {shard_count} parts")
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
            print(f"  - Completions: {tunnel.url}/v1/completions")
            print(f"  - Chat: {tunnel.url}/v1/chat/completions")
            if is_sharded:
                print(f"🔗 Serving sharded model with {shard_count} parts")
            
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
            if is_sharded:
                print(f"🔗 Using sharded model with {shard_count} parts")
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
                        timeout=120 if is_sharded else 90  # Longer timeout for sharded models
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

# --- GPU-specific functions with volumes ---
@app.function(
    gpu="T4", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_t4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="L4", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_l4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A10G", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_a10g(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="L40S", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_l40s(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A100", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_a100_40gb(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A100:8", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)  
def run_chat_a100_80gb(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="H100", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_h100(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="H200", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_h200(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_chat_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="B200", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
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

# --- Model management with volumes ---
@app.function(
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def download_model_remote(model_name: str):
    """Download a model to persistent volume"""
    print(f"📥 Downloading model to volume: {model_name}")
    model_path = get_model_path_from_name(model_name)
    download_model_to_path(model_name, model_path)
    return f"✅ Model {model_name} downloaded and saved to volume!"

@app.function(
    image=base_image,
    volumes={"/models": default_volume}
)
def list_model_files(model_name: str):
    """List files in a model's volume"""
    model_path = get_model_path_from_name(model_name)
    
    if not model_path.exists():
        return f"❌ Model {model_name} not found in volume"
    
    files = list(model_path.glob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    # Check if sharded
    is_sharded, shard_count, shard_files = check_model_sharding(model_path)
    
    result = [f"📁 Model: {model_name}"]
    result.append(f"📦 Using persistent volume storage")
    result.append(f"📁 Path: {model_path}")
    result.append(f"📊 Total size: {total_size / 1e9:.2f} GB")
    if is_sharded:
        result.append(f"🔗 Sharded model with {shard_count} parts")
    result.append(f"📄 Files ({len(files)}):")
    
    for file in sorted(files):
        if file.is_file():
            size_mb = file.stat().st_size / 1e6
            # Mark shard files
            shard_marker = " 🔗" if file.name in shard_files else ""
            result.append(f"   ✅ {file.name} ({size_mb:.1f} MB){shard_marker}")
        else:
            result.append(f"   📁 {file.name}/")
    
    return "\n".join(result)

@app.function(
    image=base_image,
    volumes={"/models": default_volume}
)
def delete_model_from_volume(model_name: str):
    """Delete a model from its volume"""
    import shutil
    
    model_path = get_model_path_from_name(model_name)
    
    if not model_path.exists():
        return f"❌ Model {model_name} not found in volume"
    
    print(f"🗑️ Deleting model {model_name} from volume...")
    shutil.rmtree(model_path)
    
    return f"✅ Model {model_name} deleted from volume"

# --- Local entrypoints ---
@app.local_entrypoint()
def chat(questions: str = ""):
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🚀 Starting chat session...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    print(f"📦 Using persistent volume for model storage")
    print(f"🔗 Sharded model support enabled")
    
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
    print(f"📦 Using persistent volume for model storage")
    print(f"🔗 Sharded model support enabled")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=None, api_only=True)

@app.local_entrypoint()
def serve_demo():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, _ = get_gpu_config(current_model)
    
    print(f"🧪 Starting API demo server (5 minutes)...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    print(f"📦 Using persistent volume for model storage")
    print(f"🔗 Sharded model support enabled")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=[], api_only=False)

@app.local_entrypoint()
def test_large_model():
    """Test large model with automatic GPU selection"""
    model_name = "unsloth/gemma-3-27b-it"
    print(f"🧪 Testing large model: {model_name}")
    print(f"🔗 Sharded model support enabled")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! I'm testing a 27B model with automatic GPU selection and sharded model support.",
        "What are the advantages of large language models?",
        "Write Python code for a binary search algorithm.",
        "Explain quantum computing in simple terms.",
        "Thank you for the demonstration!"
    ], api_only=False)

@app.local_entrypoint()
def test_sharded_model():
    """Test a known sharded model"""
    model_name = "microsoft/DialoGPT-large"  # This is often sharded
    print(f"🧪 Testing sharded model support: {model_name}")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! I'm testing sharded model loading.",
        "How are you handling multiple model files?",
        "Thank you for the demo!"
    ], api_only=False)

@app.local_entrypoint()
def test_working_model():
    """Test with a known working model"""
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"🧪 Testing known working model: {model_name}")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! Testing with a reliable 3B model.",
        "What is machine learning?",
        "Thank you!"
    ], api_only=False)

@app.local_entrypoint()
def gpu_specs():
    """Show GPU specifications and recommended models"""
    print("🚀 GPU Specifications & Model Recommendations")
    print("🔗 Now with Enhanced Sharded Model Support!")
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
    
    print("\n🔧 GPU | Memory | Model Size | Recommended Example")
    print("-" * 80)
    for gpu, memory, models, example in specs:
        print(f"{gpu:8} | {memory:7} | {models:15} | {example}")
    
    print(f"\n🔗 Enhanced Sharded Model Features:")
    print(f"  ✅ Automatic shard detection")
    print(f"  ✅ PyTorch and SafeTensors support")
    print(f"  ✅ Index file parsing")
    print(f"  ✅ Optimized loading for large models")
    print(f"  ✅ Extended timeouts for sharded models")
    print(f"  ✅ Stuck process detection and recovery")
    print(f"  ✅ Memory optimization for large shards")
    print(f"  🛠️ Compatibility fixes for vLLM 0.9.1")
    
    print(f"\n💡 Usage Examples:")
    print(f"  Working:  MODEL_NAME='Qwen/Qwen2.5-3B-Instruct' modal run script.py::serve_api")
    print(f"  Small:    MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0' modal run script.py::serve_api")
    print(f"  Medium:   MODEL_NAME='01-ai/Yi-1.5-6B-Chat' modal run script.py::serve_api")
    print(f"  Test:     modal run script.py::test_working_model")
    print(f"  Sharded:  modal run script.py::test_sharded_model")

@app.local_entrypoint()
def download(model_name: str = None):
    """Download a model to persistent volume"""
    if not model_name:
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    
    print(f"📥 Downloading model to persistent volume: {model_name}")
    print(f"🔗 Sharded models will be detected automatically")
    result = download_model_remote.remote(model_name)
    print(result)

@app.local_entrypoint()
def list_files(model_name: str = None):
    """List files in a model's volume"""
    if not model_name:
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    
    result = list_model_files.remote(model_name)
    print(result)

@app.local_entrypoint()
def delete_model(model_name: str = None):
    """Delete a model from volume"""
    if not model_name:
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    
    print(f"⚠️  Are you sure you want to delete {model_name}? This cannot be undone.")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() == 'yes':
        result = delete_model_from_volume.remote(model_name)
        print(result)
    else:
        print("❌ Deletion cancelled")

@app.local_entrypoint()
def volumes():
    """Show volume info"""
    print("📦 Modal Volume Management")
    print("🔗 With Enhanced Sharded Model Support!")
    print("=" * 50)
    
    print(f"\n📦 Volume Management:")
    print(f"   All models stored in shared persistent volume")
    print(f"   Models are cached between runs")
    print(f"   No re-downloading after first use!")
    print(f"   Sharded models fully supported")
    
    print(f"\n🔗 Enhanced Sharded Model Features:")
    print(f"   Automatic detection of model sharding")
    print(f"   Support for PyTorch (.bin) and SafeTensors")
    print(f"   Index file parsing for complex models")
    print(f"   Optimized loading for large sharded models")
    print(f"   Extended timeouts and stuck detection")
    print(f"   Compatibility fixes for vLLM 0.9.1")
    
    print(f"\n💡 Available commands:")
    print(f"   download     - Download model to volume")
    print(f"   list_files   - List files in model volume") 
    print(f"   delete_model - Delete model from volume")
    print(f"   volumes      - Show this info")
    print(f"   test_working_model - Test reliable 3B model")
    print(f"   test_sharded_model - Test sharded model support")
    
    print(f"\n🚀 Usage:")
    print(f"   modal run script.py::test_working_model")
    print(f"   modal run script.py::download --model-name 'Qwen/Qwen2.5-3B-Instruct'")
    print(f"   modal run script.py::list_files --model-name 'microsoft/DialoGPT-large'")

@app.local_entrypoint()
def info():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_type, gpu_util = get_gpu_config(current_model)
    vllm_config = get_vllm_config(current_model, gpu_type)
    
    print(f"🚀 vLLM v0.9.1 Configuration with Enhanced Sharded Model Support:")
    print(f"  Model: {current_model}")
    print(f"  GPU: {gpu_type} ({gpu_util*100}% memory)")
    print(f"  Max length: {vllm_config['max_model_len']}")
    print(f"  Tensor parallel: {vllm_config['tensor_parallel_size']}")
    print(f"  Dtype: {vllm_config['dtype']}")
    if vllm_config['quantization']:
        print(f"  Quantization: {vllm_config['quantization']}")
    
    print(f"\n📦 Volume Info:")
    print(f"  Using persistent volume storage")
    print(f"  Models persist between runs!")
    print(f"  Shared volume for all models")
    
    print(f"\n🔗 Enhanced Sharded Model Features:")
    print(f"  ✅ Automatic shard detection")
    print(f"  ✅ PyTorch (.bin) and SafeTensors support")
    print(f"  ✅ Index file parsing")
    print(f"  ✅ Optimized configuration for sharded models")
    print(f"  ✅ Extended timeouts for loading (up to 20 min)")
    print(f"  ✅ Stuck process detection and recovery")
    print(f"  ✅ Memory optimization for large shards")
    print(f"  🛠️ Compatibility fixes for vLLM 0.9.1")
    
    print(f"\n📋 Available Commands:")
    print(f"  serve_api        - Run API server indefinitely")
    print(f"  serve_demo       - Run API server for 5 minutes")
    print(f"  chat             - Run chat demo + 5 min API")
    print(f"  download         - Download model to volume")
    print(f"  list_files       - List model files in volume")
    print(f"  delete_model     - Delete model from volume")
    print(f"  volumes          - Show volume management info")
    print(f"  gpu_specs        - Show all GPU specifications")
    print(f"  test_working_model - Test reliable 3B model")
    print(f"  test_large_model - Test 27B model with auto GPU selection")
    print(f"  test_sharded_model - Test sharded model support")
    print(f"  info             - Show this configuration")
    
    print(f"\n💾 Volume Benefits:")
    print(f"  ✅ Models persist between runs - no re-downloading!")
    print(f"  ✅ Faster startup times after first download")
    print(f"  ✅ Bandwidth savings")
    print(f"  ✅ Shared volume for efficient storage")
    print(f"  🔗 Full support for sharded models of any size!")
    print(f"  🛠️ Enhanced compatibility fixes for problematic models!")
