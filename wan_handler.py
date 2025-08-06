#!/usr/bin/env python3

"""
🚀 RunPod Serverless Handler cho WAN2.2 LightX2V Q6_K - PRODUCTION FINAL VERSION
🔧 Tensor Shape Fixed, Multi-Size Image Support, CUDA Compatible
✨ Features: Enhanced Aspect Control, RIFE Interpolation, Production Error Handling
📋 Optimized for all common image sizes with comprehensive validation
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import sys
import gc
import json
import random
import traceback
import subprocess
import shutil
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
from huggingface_hub import hf_hub_download
import logging
import imageio
import numpy as np
from PIL import Image

# ================== ENHANCED CUDA COMPATIBILITY SETUP ==================
os.environ.update({
    'CUDA_LAUNCH_BLOCKING': '1',
    'TORCH_USE_CUDA_DSA': '1', 
    'CUDA_VISIBLE_DEVICES': '0',
    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True',
    'PYTORCH_KERNEL_CACHE_PATH': '/tmp/pytorch_kernel_cache',
    'CUDA_MODULE_LOADING': 'LAZY',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8'
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System paths
sys.path.insert(0, '/app/ComfyUI')
sys.path.insert(0, '/app/Practical-RIFE')

# ================== CUDA-SAFE PYTORCH CONFIGURATION ==================
def setup_pytorch_safe():
    """🔧 Setup PyTorch với CUDA safety measures"""
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.set_float32_matmul_precision('highest')
            
            try:
                device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device)
                compute_cap = torch.cuda.get_device_capability(device)
                memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
                
                logger.info(f"🎯 CUDA Device: {device_name}")
                logger.info(f"🔧 Compute Capability: {compute_cap}")
                logger.info(f"💾 Memory: {memory_gb:.1f}GB")
                
                # Test basic operations
                test_tensor = torch.ones(1, device='cuda', dtype=torch.float32)
                test_result = test_tensor + 1
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
                # Test embedding operations
                test_embed = torch.nn.Embedding(100, 32, dtype=torch.float32).cuda()
                test_input = torch.randint(0, 100, (1, 10), device='cuda')
                test_output = test_embed(test_input)
                del test_embed, test_input, test_output
                torch.cuda.empty_cache()
                
                logger.info("✅ CUDA compatibility tests passed")
                return True
                
            except RuntimeError as e:
                if "kernel image" in str(e).lower():
                    logger.error(f"❌ CUDA kernel compatibility issue: {e}")
                    torch.backends.cudnn.enabled = False
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    
                    try:
                        test_tensor = torch.ones(1, device='cuda', dtype=torch.float32)
                        del test_tensor
                        torch.cuda.empty_cache()
                        logger.info("✅ CUDA working with compatibility mode")
                        return True
                    except:
                        logger.error("❌ CUDA completely incompatible")
                        return False
                else:
                    raise e
        else:
            logger.warning("⚠️ CUDA not available, using CPU mode")
            return False
            
    except Exception as e:
        logger.error(f"❌ PyTorch setup failed: {e}")
        return False

CUDA_AVAILABLE = setup_pytorch_safe()

# ================== SAFE COMFYUI IMPORTS ==================
def safe_import_comfyui():
    """🔧 Safe ComfyUI import với error handling"""
    try:
        logger.info("📦 Importing ComfyUI modules...")
        
        from nodes import (
            CheckpointLoaderSimple, CLIPLoader, CLIPTextEncode, VAEDecode, VAELoader,
            KSampler, KSamplerAdvanced, UNETLoader, LoadImage, SaveImage,
            CLIPVisionLoader, CLIPVisionEncode, LoraLoaderModelOnly, ImageScale
        )
        
        from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
        
        from custom_nodes.ComfyUI_KJNodes.nodes.model_optimization_nodes import (
            WanVideoTeaCacheKJ, PathchSageAttentionKJ, WanVideoNAG, SkipLayerGuidanceWanVideo
        )
        
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes_wan import WanImageToVideo
        from comfy import model_management
        import folder_paths
        
        logger.info("✅ ComfyUI modules imported successfully")
        
        return True, {
            'CLIPLoader': CLIPLoader,
            'CLIPTextEncode': CLIPTextEncode,
            'VAEDecode': VAEDecode,
            'VAELoader': VAELoader,
            'UnetLoaderGGUF': UnetLoaderGGUF,
            'LoadImage': LoadImage,
            'ImageScale': ImageScale,
            'WanImageToVideo': WanImageToVideo,
            'KSamplerAdvanced': KSamplerAdvanced,
            'LoraLoaderModelOnly': LoraLoaderModelOnly,
            'PathchSageAttentionKJ': PathchSageAttentionKJ,
            'WanVideoTeaCacheKJ': WanVideoTeaCacheKJ,
            'WanVideoNAG': WanVideoNAG,
            'ModelSamplingSD3': ModelSamplingSD3,
            'CLIPVisionLoader': CLIPVisionLoader,
            'CLIPVisionEncode': CLIPVisionEncode
        }
        
    except ImportError as e:
        logger.error(f"❌ ComfyUI import failed: {e}")
        return False, {}

COMFYUI_AVAILABLE, COMFYUI_NODES = safe_import_comfyui()

# ================== MODEL CONFIGURATIONS ==================
MODEL_CONFIGS = {
    "dit_model_high": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
    "dit_model_low": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf",
    "text_encoder": "/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "/app/ComfyUI/models/vae/wan_2.1_vae.safetensors",
    "clip_vision": "/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
    "lightx2v_rank_32": "/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
    "lightx2v_rank_64": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
    "lightx2v_rank_128": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
    "walking_to_viewers": "/app/ComfyUI/models/loras/walking to viewers_Wan.safetensors",
    "walking_from_behind": "/app/ComfyUI/models/loras/walking_from_behind.safetensors",
    "dancing": "/app/ComfyUI/models/loras/b3ll13-d8nc3r.safetensors",
    "pusa_lora": "/app/ComfyUI/models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors"
}

# ================== MINIO CONFIGURATION ==================
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    logger.info("✅ MinIO client initialized")
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# ================== UTILITY FUNCTIONS ==================

def clear_memory_enhanced():
    """🧹 Enhanced memory cleanup"""
    try:
        gc.collect()
        if CUDA_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup warning: {e}")

def verify_models() -> tuple[bool, list]:
    """🔍 Verify all required models exist"""
    logger.info("🔍 Verifying all required models...")
    missing_models = []
    existing_models = []
    total_size = 0
    
    for name, path in MODEL_CONFIGS.items():
        if os.path.exists(path):
            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                existing_models.append(f"{name}: {file_size_mb:.1f}MB")
                total_size += file_size_mb
                logger.info(f"✅ {name}: {file_size_mb:.1f}MB")
            except Exception as e:
                logger.error(f"❌ Error checking {name}: {e}")
                missing_models.append(f"{name}: {path} (error)")
        else:
            missing_models.append(f"{name}: {path}")
            logger.error(f"❌ Missing: {name}")
    
    if missing_models:
        logger.error(f"❌ Missing {len(missing_models)}/{len(MODEL_CONFIGS)} models")
        return False, missing_models
    else:
        logger.info(f"✅ All {len(existing_models)} models verified! Total: {total_size:.1f}MB")
        return True, []

def get_lightx2v_lora_path(lightx2v_rank: str) -> str:
    """📍 Get LightX2V LoRA path by rank"""
    rank_mapping = {
        "32": "lightx2v_rank_32",
        "64": "lightx2v_rank_64", 
        "128": "lightx2v_rank_128"
    }
    config_key = rank_mapping.get(lightx2v_rank, "lightx2v_rank_32")
    return MODEL_CONFIGS.get(config_key)

def download_lora_dynamic(lora_url: str, civitai_token: str = None) -> str:
    """🎨 Download LoRA from various sources"""
    try:
        lora_dir = "/app/ComfyUI/models/loras"
        os.makedirs(lora_dir, exist_ok=True)
        
        logger.info(f"🎨 Downloading LoRA from: {lora_url}")
        
        if "huggingface.co" in lora_url:
            parts = lora_url.split("/")
            if len(parts) >= 7 and "resolve" in parts:
                username = parts[3]
                repo = parts[4]
                filename = parts[-1]
                
                logger.info(f"📥 HuggingFace: {username}/{repo}/{filename}")
                downloaded_path = hf_hub_download(
                    repo_id=f"{username}/{repo}",
                    filename=filename,
                    local_dir=lora_dir,
                    force_download=True
                )
                logger.info(f"✅ HuggingFace LoRA downloaded: {filename}")
                return downloaded_path
                
        elif "civitai.com" in lora_url:
            try:
                model_id = lora_url.split("/models/")[1].split("?")[0].split("/")[0]
                headers = {}
                if civitai_token:
                    headers["Authorization"] = f"Bearer {civitai_token}"
                
                api_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
                response = requests.get(api_url, headers=headers, timeout=300, stream=True)
                response.raise_for_status()
                
                timestamp = int(time.time())
                filename = f"civitai_model_{model_id}_{timestamp}.safetensors"
                local_path = os.path.join(lora_dir, filename)
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = os.path.getsize(local_path) / (1024 * 1024)
                logger.info(f"✅ CivitAI LoRA downloaded: {filename} ({file_size:.1f}MB)")
                return local_path
                
            except Exception as e:
                logger.error(f"❌ CivitAI download failed: {e}")
                return None
        else:
            # Direct download
            filename = os.path.basename(urlparse(lora_url).path)
            if not filename.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                filename = f"downloaded_lora_{int(time.time())}.safetensors"
            
            local_path = os.path.join(lora_dir, filename)
            response = requests.get(lora_url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"✅ Direct LoRA downloaded: {filename} ({file_size:.1f}MB)")
            return local_path
        
        return None
        
    except Exception as e:
        logger.error(f"❌ LoRA download failed: {e}")
        return None

# ================== ASPECT RATIO PROCESSING (FIXED) ==================

def calculate_aspect_preserving_dimensions(original_width, original_height, target_width, target_height):
    """📐 Calculate dimensions preserving aspect ratio với padding"""
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height
    
    if original_ratio > target_ratio:
        new_width = target_width
        new_height = int(target_width / original_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * original_ratio)
    
    # Ensure divisible by 8 for video encoding
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height

def calculate_crop_dimensions(original_width, original_height, target_width, target_height):
    """✂️ Calculate smart crop dimensions"""
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    
    if original_ratio > target_ratio:
        crop_height = original_height
        crop_width = int(original_height * target_ratio)
        crop_x = (original_width - crop_width) // 2
        crop_y = 0
    else:
        crop_width = original_width
        crop_height = int(original_width / target_ratio)
        crop_x = 0
        crop_y = (original_height - crop_height) // 2
    
    return crop_x, crop_y, crop_width, crop_height

def process_image_with_aspect_control_fixed(
    image_path: str,
    target_width: int = 720,
    target_height: int = 1280,
    aspect_mode: str = "preserve",
    auto_720p: bool = False
):
    """
    🔧 FIXED: Enhanced image processing với comprehensive tensor validation
    ✅ Handles all common image sizes correctly
    """
    try:
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI not available")
        
        load_image = COMFYUI_NODES['LoadImage']()
        image_scaler = COMFYUI_NODES['ImageScale']()
        
        # Load original image
        loaded_image = load_image.load_image(image_path)[0]
        
        # Enhanced tensor validation
        logger.info(f"📊 Initial loaded_image shape: {loaded_image.shape}")
        logger.info(f"📊 Initial loaded_image dtype: {loaded_image.dtype}")
        
        # Get original dimensions với enhanced validation
        if loaded_image.ndim == 4:
            batch, orig_height, orig_width, channels = loaded_image.shape
            if batch != 1:
                logger.warning(f"⚠️ Unexpected batch size: {batch}, taking first item")
                loaded_image = loaded_image[0:1]
        elif loaded_image.ndim == 3:
            orig_height, orig_width, channels = loaded_image.shape
            loaded_image = loaded_image.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Unexpected image tensor shape: {loaded_image.shape}")
        
        logger.info(f"🖼️ Original image: {orig_width}x{orig_height} (channels: {channels})")
        logger.info(f"📊 Normalized tensor shape: {loaded_image.shape}")
        
        # Auto 720p detection
        if auto_720p:
            if orig_width > orig_height:
                target_width, target_height = 1280, 720
            else:
                target_width, target_height = 720, 1280
            logger.info(f"📱 Auto 720p mode: {target_width}x{target_height}")
        
        # Process based on aspect mode với proper tensor handling
        if aspect_mode == "preserve":
            # Calculate new dimensions
            new_width, new_height = calculate_aspect_preserving_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            logger.info(f"📐 Calculated dimensions: {new_width}x{new_height}")
            
            # Scale to calculated size first
            scaled_image = image_scaler.upscale(
                loaded_image, "lanczos", new_width, new_height, "disabled"
            )[0]
            
            logger.info(f"📊 Scaled image shape: {scaled_image.shape}")
            
            # Add padding if needed
            if new_width != target_width or new_height != target_height:
                logger.info(f"🔧 Adding padding: {new_width}x{new_height} -> {target_width}x{target_height}")
                
                # Calculate padding values
                pad_x = (target_width - new_width) // 2
                pad_y = (target_height - new_height) // 2
                pad_x_right = target_width - new_width - pad_x
                pad_y_bottom = target_height - new_height - pad_y
                
                logger.info(f"🔧 Padding values: left={pad_x}, right={pad_x_right}, top={pad_y}, bottom={pad_y_bottom}")
                
                # Ensure proper tensor format for padding
                if scaled_image.ndim == 3:
                    scaled_image = scaled_image.unsqueeze(0)
                
                logger.info(f"📊 Pre-padding shape: {scaled_image.shape}")
                
                # Convert BHWC -> BCHW for F.pad
                scaled_image_bchw = scaled_image.permute(0, 3, 1, 2)
                logger.info(f"📊 BCHW format shape: {scaled_image_bchw.shape}")
                
                # Apply padding: (left, right, top, bottom)
                padded_bchw = F.pad(
                    scaled_image_bchw,
                    (pad_x, pad_x_right, pad_y, pad_y_bottom),
                    mode='constant',
                    value=0
                )
                
                logger.info(f"📊 Padded BCHW shape: {padded_bchw.shape}")
                
                # Convert back BCHW -> BHWC và remove batch
                final_image = padded_bchw.permute(0, 2, 3, 1)[0]
                
                logger.info(f"📊 Final padded shape: {final_image.shape}")
                
            else:
                # Remove batch dimension if present
                if scaled_image.ndim == 4:
                    final_image = scaled_image[0]
                else:
                    final_image = scaled_image
                    
            logger.info(f"📐 Preserve mode completed: {orig_width}x{orig_height} -> {target_width}x{target_height}")
            
        elif aspect_mode == "crop":
            # Smart crop implementation
            crop_x, crop_y, crop_width, crop_height = calculate_crop_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            logger.info(f"✂️ Crop parameters: x={crop_x}, y={crop_y}, w={crop_width}, h={crop_height}")
            
            # Ensure proper batch handling for cropping
            if loaded_image.ndim == 4:
                cropped = loaded_image[0, crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
                cropped = cropped.unsqueeze(0)  # Re-add batch for upscale
            else:
                cropped = loaded_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
                cropped = cropped.unsqueeze(0)
            
            logger.info(f"📊 Cropped shape: {cropped.shape}")
            
            # Scale to target size
            final_image = image_scaler.upscale(
                cropped, "lanczos", target_width, target_height, "disabled"
            )[0]
            
            logger.info(f"✂️ Crop mode completed: {orig_width}x{orig_height} -> {target_width}x{target_height}")
            
        else:  # stretch
            # Direct stretch - simplest approach
            final_image = image_scaler.upscale(
                loaded_image, "lanczos", target_width, target_height, "disabled"
            )[0]
            
            logger.info(f"🔄 Stretch mode completed: {orig_width}x{orig_height} -> {target_width}x{target_height}")
        
        # FINAL VALIDATION
        logger.info(f"📊 Final image validation:")
        logger.info(f"  Shape: {final_image.shape}")
        logger.info(f"  Dtype: {final_image.dtype}")
        logger.info(f"  Expected: [H={target_height}, W={target_width}, C=3]")
        
        # Validate final tensor
        if final_image.ndim != 3:
            raise ValueError(f"Final image must be 3D [H,W,C], got {final_image.ndim}D: {final_image.shape}")
        
        actual_height, actual_width, actual_channels = final_image.shape
        
        if actual_height != target_height or actual_width != target_width:
            logger.error(f"❌ Size mismatch: expected {target_height}x{target_width}, got {actual_height}x{actual_width}")
            raise ValueError(f"Size mismatch: expected {target_height}x{target_width}, got {actual_height}x{actual_width}")
        
        if actual_channels not in [1, 3, 4]:
            raise ValueError(f"Invalid channel count: {actual_channels}")
        
        # Ensure RGB format
        if actual_channels == 1:
            logger.info("🔄 Converting grayscale to RGB...")
            final_image = final_image.repeat(1, 1, 3)
        elif actual_channels == 4:
            logger.info("🔄 Removing alpha channel...")
            final_image = final_image[:, :, :3]
        
        logger.info(f"✅ Image processing completed successfully!")
        logger.info(f"📊 Final output: {final_image.shape}")
        
        return final_image, target_width, target_height
        
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

# ================== CUDA-SAFE TEXT PROCESSING ==================

def safe_load_text_encoder():
    """🔧 CUDA-safe text encoder loading"""
    try:
        clear_memory_enhanced()
        
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI not available")
        
        clip_loader = COMFYUI_NODES['CLIPLoader']()
        
        try:
            logger.info("📝 Loading text encoder...")
            clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            
            # Test basic functionality
            clip_encode = COMFYUI_NODES['CLIPTextEncode']()
            test_result = clip_encode.encode(clip, "test prompt")
            del test_result
            
            logger.info("✅ Text encoder loaded successfully")
            return clip
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "cuda error" in error_str or "kernel image" in error_str:
                logger.warning("⚠️ CUDA compatibility issue, applying fixes...")
                
                # Disable CUDNN và retry
                original_cudnn = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                
                try:
                    clear_memory_enhanced()
                    clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
                    logger.info("✅ Text encoder loaded with CUDNN disabled")
                    return clip
                except Exception:
                    torch.backends.cudnn.enabled = original_cudnn
                    pass
                
                # CPU fallback
                logger.warning("🔄 Attempting CPU fallback...")
                try:
                    original_cuda_available = torch.cuda.is_available
                    torch.cuda.is_available = lambda: False
                    
                    clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
                    torch.cuda.is_available = original_cuda_available
                    
                    logger.info("✅ Text encoder loaded in CPU mode")
                    return clip
                except Exception:
                    torch.cuda.is_available = original_cuda_available
                    raise e
            else:
                raise e
                
    except Exception as e:
        logger.error(f"❌ Text encoder loading failed: {e}")
        raise e

def safe_encode_text(clip, text):
    """🔧 CUDA-safe text encoding"""
    try:
        clip_encode = COMFYUI_NODES['CLIPTextEncode']()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = clip_encode.encode(clip, text)
                return result[0]
            except RuntimeError as e:
                error_str = str(e).lower()
                if ("cuda error" in error_str or "kernel" in error_str) and attempt < max_retries - 1:
                    logger.warning(f"⚠️ CUDA error on encoding attempt {attempt + 1}, retrying...")
                    clear_memory_enhanced()
                    time.sleep(1)
                    continue
                else:
                    raise e
                    
    except Exception as e:
        logger.error(f"❌ Text encoding failed: {e}")
        raise e

# ================== RIFE INTERPOLATION (ULTIMATE FIXED) ==================

def apply_rife_interpolation_ultimate_fixed(video_path: str, interpolation_factor: int = 2) -> str:
    """🔧 ULTIMATE FIXED RIFE interpolation với comprehensive output detection"""
    try:
        logger.info(f"🔄 Applying RIFE interpolation (factor: {interpolation_factor}) - ULTIMATE FIXED...")
        
        if not os.path.exists(video_path):
            logger.error(f"❌ Input video not found: {video_path}")
            return video_path
        
        original_size = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"📊 Original video: {original_size:.1f}MB")
        
        # Check RIFE script
        rife_script = "/app/Practical-RIFE/inference_video.py"
        if not os.path.exists(rife_script):
            logger.error(f"❌ RIFE script not found")
            return video_path
        
        # Setup directories
        rife_output_dir = "/app/rife_output"
        os.makedirs(rife_output_dir, exist_ok=True)
        
        # Generate filenames
        input_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{input_basename}_rife_{interpolation_factor}x.mp4"
        final_output_path = os.path.join(rife_output_dir, output_filename)
        
        # Copy input to RIFE directory
        rife_input_path = os.path.join("/app/Practical-RIFE", os.path.basename(video_path))
        shutil.copy2(video_path, rife_input_path)
        
        # Enhanced environment
        env = os.environ.copy()
        env.update({
            "XDG_RUNTIME_DIR": "/tmp",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "FFMPEG_LOGLEVEL": "quiet",
            "CUDA_VISIBLE_DEVICES": "0"
        })
        
        # Build command
        cmd = [
            "python3",
            "inference_video.py",
            f"--multi={interpolation_factor}",
            f"--video={os.path.basename(video_path)}",
            "--scale=1.0",
            "--fps=30",
            "--png=False"
        ]
        
        logger.info(f"🔧 RIFE command: {' '.join(cmd)}")
        
        # Execute RIFE
        original_cwd = os.getcwd()
        os.chdir("/app/Practical-RIFE")
        
        try:
            # Clear previous outputs
            cleanup_patterns = ["*_interpolated.mp4", "*_rife.mp4", "*X.mp4"]
            for pattern in cleanup_patterns:
                for file in Path(".").glob(pattern):
                    try:
                        file.unlink()
                    except:
                        pass
            
            logger.info("🚀 Starting RIFE interpolation...")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minutes
                check=False
            )
            
            execution_time = time.time() - start_time
            logger.info(f"⏱️ RIFE execution time: {execution_time:.1f}s")
            logger.info(f"📊 Return code: {result.returncode}")
            
        except subprocess.TimeoutExpired:
            logger.error("❌ RIFE timed out")
            return video_path
        except Exception as e:
            logger.error(f"❌ RIFE execution failed: {e}")
            return video_path
        finally:
            os.chdir(original_cwd)
            try:
                if os.path.exists(rife_input_path):
                    os.remove(rife_input_path)
            except:
                pass
        
        # ULTIMATE OUTPUT DETECTION
        logger.info("🔍 Ultimate output detection...")
        
        potential_outputs = []
        rife_dir = "/app/Practical-RIFE"
        
        # Comprehensive search patterns
        search_patterns = [
            f"{input_basename}_{interpolation_factor}X.mp4",
            f"{input_basename}_{interpolation_factor}x.mp4",
            f"{input_basename}_interpolated.mp4",
            f"output_{interpolation_factor}x.mp4",
            "output.mp4"
        ]
        
        # Search by patterns
        for pattern in search_patterns:
            potential_path = os.path.join(rife_dir, pattern)
            if os.path.exists(potential_path):
                potential_outputs.append(potential_path)
                logger.info(f"🔍 Found: {pattern}")
        
        # Search recent files
        current_time = time.time()
        try:
            for file in os.listdir(rife_dir):
                if file.endswith('.mp4') and file != os.path.basename(video_path):
                    file_path = os.path.join(rife_dir, file)
                    if current_time - max(os.path.getctime(file_path), os.path.getmtime(file_path)) < 1200:
                        if file_path not in potential_outputs:
                            potential_outputs.append(file_path)
        except Exception as e:
            logger.warning(f"⚠️ Directory scan error: {e}")
        
        # Select best output
        selected_output = None
        if potential_outputs:
            logger.info(f"📊 Analyzing {len(potential_outputs)} potential outputs")
            
            # Sort by size (larger is likely better for interpolated video)
            candidates = []
            for path in potential_outputs:
                try:
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    candidates.append({'path': path, 'size_mb': size_mb})
                except:
                    continue
            
            candidates.sort(key=lambda x: x['size_mb'], reverse=True)
            
            for candidate in candidates:
                # Validate candidate meets requirements
                if candidate['size_mb'] >= original_size * 0.3:  # At least 30% of original
                    selected_output = candidate['path']
                    logger.info(f"✅ Selected: {os.path.basename(selected_output)} ({candidate['size_mb']:.1f}MB)")
                    break
        
        # Process selected output
        if selected_output and os.path.exists(selected_output):
            try:
                shutil.move(selected_output, final_output_path)
                
                final_size = os.path.getsize(final_output_path) / (1024 * 1024)
                logger.info(f"✅ RIFE interpolation successful!")
                logger.info(f"📊 {original_size:.1f}MB → {final_size:.1f}MB")
                
                return final_output_path
                
            except Exception as e:
                logger.error(f"❌ Failed to move output: {e}")
                return video_path
        else:
            logger.warning("⚠️ No valid output found, using original")
            return video_path
            
    except Exception as e:
        logger.error(f"❌ RIFE interpolation error: {e}")
        return video_path

# ================== VIDEO SAVING ==================

def save_video_production(frames_tensor, output_path, fps=16):
    """🎬 Production-grade video saving"""
    try:
        logger.info(f"🎬 Saving video: {output_path}")
        
        if frames_tensor is None:
            raise ValueError("Frames tensor is None")
        
        # Convert to numpy
        if torch.is_tensor(frames_tensor):
            frames_np = frames_tensor.detach().cpu().float().numpy()
        else:
            frames_np = np.array(frames_tensor, dtype=np.float32)
        
        logger.info(f"📊 Input shape: {frames_np.shape}")
        
        # Handle batch dimension
        if frames_np.ndim == 5 and frames_np.shape[0] == 1:
            frames_np = frames_np[0]
        
        # Convert to uint8
        if frames_np.dtype != np.uint8:
            if frames_np.max() <= 1.0:
                frames_np = (frames_np * 255.0).astype(np.uint8)
            else:
                frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)
        
        # Validate dimensions
        if len(frames_np.shape) != 4:
            raise ValueError(f"Invalid shape: {frames_np.shape} (expected 4D)")
        
        num_frames, height, width, channels = frames_np.shape
        
        # Handle channels
        if channels == 1:
            logger.info("🔄 Converting grayscale to RGB...")
            frames_np = np.repeat(frames_np, 3, axis=-1)
            channels = 3
        elif channels == 4:
            logger.info("🔄 Removing alpha channel...")
            frames_np = frames_np[:, :, :, :3]
            channels = 3
        elif channels != 3:
            raise ValueError(f"Unsupported channels: {channels}")
        
        logger.info(f"📊 Final specs: {num_frames} frames, {width}x{height}, {channels} channels")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Multiple encoding strategies
        strategies = [
            {
                'name': 'High Quality',
                'params': {
                    'fps': fps,
                    'codec': 'libx264',
                    'pixelformat': 'yuv420p',
                    'quality': 8
                }
            },
            {
                'name': 'Standard',
                'params': {'fps': fps, 'codec': 'libx264', 'pixelformat': 'yuv420p'}
            },
            {
                'name': 'Basic',
                'params': {'fps': fps}
            }
        ]
        
        # Try strategies
        for strategy in strategies:
            try:
                logger.info(f"🎥 Trying {strategy['name']} encoding...")
                
                with imageio.get_writer(output_path, **strategy['params']) as writer:
                    for i, frame in enumerate(frames_np):
                        writer.append_data(frame)
                        if (i + 1) % max(1, num_frames // 10) == 0:
                            progress = ((i + 1) / num_frames) * 100
                            logger.info(f"📹 Progress: {progress:.1f}%")
                
                # Verify output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"✅ Video saved with {strategy['name']} encoding!")
                    logger.info(f"📊 Stats: {file_size_mb:.2f}MB, {num_frames} frames @ {fps}fps")
                    return output_path
                else:
                    logger.warning(f"⚠️ {strategy['name']} produced empty file")
                    
            except Exception as e:
                logger.warning(f"⚠️ {strategy['name']} failed: {e}")
                continue
        
        raise RuntimeError("All encoding strategies failed")
        
    except Exception as e:
        logger.error(f"❌ Video saving failed: {e}")
        raise e

# ================== MAIN VIDEO GENERATION ==================

def generate_video_wan22_complete_fixed(image_path: str, **kwargs) -> str:
    """
    🎬 COMPLETE FIXED WAN2.2 video generation
    ✅ Multi-size image support, tensor shape fixes, comprehensive error handling
    """
    try:
        logger.info("🎬 Starting WAN2.2 COMPLETE FIXED generation...")
        
        # Extract parameters với defaults
        positive_prompt = kwargs.get('positive_prompt', '')
        negative_prompt = kwargs.get('negative_prompt', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
        
        aspect_mode = kwargs.get('aspect_mode', 'preserve')
        auto_720p = kwargs.get('auto_720p', False)
        width = kwargs.get('width', 720)
        height = kwargs.get('height', 1280)
        seed = kwargs.get('seed', 0)
        steps = kwargs.get('steps', 6)
        high_noise_steps = kwargs.get('high_noise_steps', 3)
        cfg_scale = kwargs.get('cfg_scale', 1.0)
        sampler_name = kwargs.get('sampler_name', 'euler')
        scheduler = kwargs.get('scheduler', 'simple')
        frames = kwargs.get('frames', 65)
        fps = kwargs.get('fps', 16)
        
        # Advanced features
        prompt_assist = kwargs.get('prompt_assist', 'none')
        use_lightx2v = kwargs.get('use_lightx2v', True)
        lightx2v_rank = kwargs.get('lightx2v_rank', '32')
        lightx2v_strength = kwargs.get('lightx2v_strength', 3.0)
        use_pusa = kwargs.get('use_pusa', True)
        pusa_strength = kwargs.get('pusa_strength', 1.5)
        
        # LoRA settings
        use_lora = kwargs.get('use_lora', False)
        lora_url = kwargs.get('lora_url', None)
        lora_strength = kwargs.get('lora_strength', 1.0)
        civitai_token = kwargs.get('civitai_token', None)
        
        # Optimizations
        use_sage_attention = kwargs.get('use_sage_attention', True)
        rel_l1_thresh = kwargs.get('rel_l1_thresh', 0.0)
        
        # Frame interpolation
        enable_interpolation = kwargs.get('enable_interpolation', False)
        interpolation_factor = kwargs.get('interpolation_factor', 2)
        
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
        
        logger.info(f"🎯 Configuration:")
        logger.info(f"  Resolution: {width}x{height} (mode: {aspect_mode})")
        logger.info(f"  Animation: {frames} frames @ {fps}fps")
        logger.info(f"  Sampling: {steps} steps, CFG: {cfg_scale}, seed: {seed}")
        logger.info(f"  LightX2V: rank {lightx2v_rank}, strength {lightx2v_strength}")
        logger.info(f"  Interpolation: {enable_interpolation} (factor: {interpolation_factor})")
        
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI not available")
        
        with torch.inference_mode():
            # Initialize nodes
            logger.info("🔧 Initializing nodes...")
            unet_loader = COMFYUI_NODES['UnetLoaderGGUF']()
            vae_loader = COMFYUI_NODES['VAELoader']()
            wan_image_to_video = COMFYUI_NODES['WanImageToVideo']()
            ksampler = COMFYUI_NODES['KSamplerAdvanced']()
            vae_decode = COMFYUI_NODES['VAEDecode']()
            
            # LoRA loaders
            pAssLora = COMFYUI_NODES['LoraLoaderModelOnly']()
            load_lora_node = COMFYUI_NODES['LoraLoaderModelOnly']()
            load_lightx2v_lora = COMFYUI_NODES['LoraLoaderModelOnly']()
            load_pusa_lora = COMFYUI_NODES['LoraLoaderModelOnly']()
            
            # Optimizations (if available)
            pathch_sage_attention = COMFYUI_NODES.get('PathchSageAttentionKJ')
            teacache = COMFYUI_NODES.get('WanVideoTeaCacheKJ')
            
            # Text encoding với CUDA safety
            logger.info("📝 Loading text encoder safely...")
            clip = safe_load_text_encoder()
            
            # Process prompts
            final_positive_prompt = positive_prompt
            if prompt_assist != "none":
                final_positive_prompt = f"{positive_prompt} {prompt_assist}."
            
            positive = safe_encode_text(clip, final_positive_prompt)
            negative = safe_encode_text(clip, negative_prompt)
            
            del clip
            clear_memory_enhanced()
            
            # FIXED: Image processing với comprehensive validation
            logger.info("🖼️ Processing image with FIXED aspect control...")
            loaded_image, final_width, final_height = process_image_with_aspect_control_fixed(
                image_path=image_path,
                target_width=width,
                target_height=height,
                aspect_mode=aspect_mode,
                auto_720p=auto_720p
            )
            
            width, height = final_width, final_height
            
            # CRITICAL: Validate tensor before wan_image_to_video
            logger.info("🔍 Pre-encode validation:")
            logger.info(f"  Image shape: {loaded_image.shape}")
            logger.info(f"  Target size: {width}x{height}")
            
            if loaded_image.ndim != 3:
                raise ValueError(f"Image must be 3D [H,W,C], got {loaded_image.ndim}D")
            
            actual_h, actual_w, actual_c = loaded_image.shape
            if actual_h != height or actual_w != width:
                raise ValueError(f"Size mismatch: {actual_h}x{actual_w} vs expected {height}x{width}")
            
            # Load VAE
            logger.info("🎨 Loading VAE...")
            vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            
            # FIXED: Image to video encoding với proper validation
            logger.info("🔄 Encoding image to video latent...")
            try:
                positive_out, negative_out, latent = wan_image_to_video.encode(
                    positive, negative, vae, width, height, frames, 1, loaded_image, None
                )
                logger.info("✅ Image to video encoding successful!")
            except Exception as e:
                logger.error(f"❌ Encoding failed: {e}")
                logger.error(f"Debug: image={loaded_image.shape}, size={width}x{height}")
                raise e
            
            # STAGE 1: High noise model
            logger.info("🎯 STAGE 1: High noise sampling...")
            model = unet_loader.load_unet("wan2.2_i2v_high_noise_14B_Q6_K.gguf")[0]
            
            # Apply prompt assist LoRAs
            if prompt_assist != "none":
                lora_mapping = {
                    "walking to viewers": "walking to viewers_Wan.safetensors",
                    "walking from behind": "walking_from_behind.safetensors", 
                    "b3ll13-d8nc3r": "b3ll13-d8nc3r.safetensors"
                }
                
                if prompt_assist in lora_mapping:
                    lora_file = lora_mapping[prompt_assist]
                    logger.info(f"🎭 Applying LoRA: {lora_file}")
                    model = pAssLora.load_lora_model_only(model, lora_file, 1.0)[0]
            
            # Apply custom LoRA
            if use_lora and lora_url:
                logger.info("🎨 Processing custom LoRA...")
                lora_path = download_lora_dynamic(lora_url, civitai_token)
                if lora_path:
                    model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), lora_strength)[0]
                    logger.info(f"✅ Custom LoRA applied")
            
            # Apply LightX2V LoRA
            if use_lightx2v:
                logger.info(f"⚡ Loading LightX2V LoRA rank {lightx2v_rank}...")
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path and os.path.exists(lightx2v_lora_path):
                    model = load_lightx2v_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), lightx2v_strength
                    )[0]
                    logger.info("✅ LightX2V LoRA applied")
            
            # Apply optimizations
            if use_sage_attention and pathch_sage_attention:
                try:
                    model = pathch_sage_attention().patch(model, "auto")[0]
                    logger.info("✅ Sage Attention applied")
                except Exception as e:
                    logger.warning(f"⚠️ Sage Attention failed: {e}")
            
            if rel_l1_thresh > 0 and teacache:
                try:
                    model = teacache().patch_teacache(model, rel_l1_thresh, 0.2, 1.0, "main_device", "14B")[0]
                    logger.info(f"✅ TeaCache applied")
                except Exception as e:
                    logger.warning(f"⚠️ TeaCache failed: {e}")
            
            # Sample với high noise model
            sampled = ksampler.sample(
                model=model,
                add_noise="enable",
                noise_seed=seed,
                steps=steps,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_out,
                negative=negative_out,
                latent_image=latent,
                start_at_step=0,
                end_at_step=high_noise_steps,
                return_with_leftover_noise="enable"
            )[0]
            
            del model
            clear_memory_enhanced()
            
            # STAGE 2: Low noise model
            logger.info("🎯 STAGE 2: Low noise sampling...")
            model = unet_loader.load_unet("wan2.2_i2v_low_noise_14B_Q6_K.gguf")[0]
            
            # Reapply LoRAs for low noise model
            if prompt_assist != "none" and prompt_assist in lora_mapping:
                model = pAssLora.load_lora_model_only(model, lora_mapping[prompt_assist], 1.0)[0]
            
            if use_lora and lora_url and lora_path:
                model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), lora_strength)[0]
            
            # Apply PUSA LoRA
            if use_pusa:
                logger.info(f"🎭 Loading PUSA LoRA...")
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path and os.path.exists(lightx2v_lora_path):
                    model = load_pusa_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), pusa_strength
                    )[0]
                    logger.info("✅ PUSA LoRA applied")
            
            # Reapply optimizations
            if use_sage_attention and pathch_sage_attention:
                try:
                    model = pathch_sage_attention().patch(model, "auto")[0]
                except Exception as e:
                    logger.warning(f"⚠️ Sage Attention (stage 2) failed: {e}")
            
            # Sample với low noise model
            sampled = ksampler.sample(
                model=model,
                add_noise="disable",
                noise_seed=seed,
                steps=steps,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_out,
                negative=negative_out,
                latent_image=sampled,
                start_at_step=high_noise_steps,
                end_at_step=10000,
                return_with_leftover_noise="disable"
            )[0]
            
            del model
            clear_memory_enhanced()
            
            # Decode frames
            logger.info("🎨 Decoding latents...")
            decoded = vae_decode.decode(vae, sampled)[0]
            
            del vae
            clear_memory_enhanced()
            
            # Save video
            logger.info("💾 Saving video...")
            output_path = f"/app/ComfyUI/output/wan22_complete_fixed_{uuid.uuid4().hex[:8]}.mp4"
            
            final_output_path = save_video_production(decoded, output_path, fps)
            
            # Frame interpolation
            if enable_interpolation and interpolation_factor > 1:
                logger.info(f"🔄 Applying ULTIMATE FIXED RIFE interpolation...")
                
                pre_interp_size = os.path.getsize(final_output_path) / (1024 * 1024)
                logger.info(f"📊 Pre-interpolation: {pre_interp_size:.1f}MB")
                
                interpolated_path = apply_rife_interpolation_ultimate_fixed(final_output_path, interpolation_factor)
                
                if interpolated_path != final_output_path and os.path.exists(interpolated_path):
                    post_interp_size = os.path.getsize(interpolated_path) / (1024 * 1024)
                    logger.info(f"📊 Post-interpolation: {post_interp_size:.1f}MB")
                    logger.info(f"✅ Frame interpolation successful!")
                    return interpolated_path
                else:
                    logger.warning("⚠️ Frame interpolation failed, using original")
                    return final_output_path
            
            return final_output_path
            
    except Exception as e:
        logger.error(f"❌ Video generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        clear_memory_enhanced()

# ================== UPLOAD FUNCTIONS ==================

def upload_to_minio(local_path: str, object_name: str) -> str:
    """📤 Upload file to MinIO storage"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not available")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading to MinIO: {object_name} ({file_size_mb:.1f}MB)")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Upload completed: {file_url}")
        
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

# ================== INPUT VALIDATION ==================

def validate_input_comprehensive(job_input: dict) -> tuple[bool, str]:
    """🔍 Comprehensive input validation với multi-size support"""
    try:
        # Required parameters
        required_params = ["image_url", "positive_prompt"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        # Image URL validation
        image_url = job_input["image_url"]
        try:
            response = requests.head(image_url, timeout=10)
            if response.status_code not in [200, 301, 302]:
                return False, f"Image URL not accessible: {response.status_code}"
        except Exception as e:
            return False, f"Image URL validation failed: {str(e)}"
        
        # Validate dimensions - ENHANCED for multi-size support
        width = job_input.get("width", 720)
        height = job_input.get("height", 1280)
        if not (256 <= width <= 1536 and 256 <= height <= 1536):
            return False, "Width and height must be between 256 and 1536"
        
        # Validate frames
        frames = job_input.get("frames", 65)
        if not (1 <= frames <= 150):
            return False, "Frames must be between 1 and 150"
        
        # Enhanced prompt_assist validation
        prompt_assist = job_input.get("prompt_assist", "none")
        if isinstance(prompt_assist, str):
            prompt_assist_normalized = prompt_assist.lower().strip()
        else:
            prompt_assist_normalized = str(prompt_assist).lower().strip()
        
        valid_assists_map = {
            "none": "none",
            "walking to viewers": "walking to viewers",
            "walking from behind": "walking from behind", 
            "b3ll13-d8nc3r": "b3ll13-d8nc3r",
            "walking_to_viewers": "walking to viewers",
            "walking_from_behind": "walking from behind",
            "dance": "b3ll13-d8nc3r",
            "dancing": "b3ll13-d8nc3r",
            "dancer": "b3ll13-d8nc3r",
            "": "none",
            "null": "none"
        }
        
        if prompt_assist_normalized not in valid_assists_map:
            valid_display = ["none", "walking to viewers", "walking from behind", "b3ll13-d8nc3r"]
            return False, f"prompt_assist must be one of: {', '.join(valid_display)}"
        
        job_input["prompt_assist"] = valid_assists_map[prompt_assist_normalized]
        
        # Validate aspect mode
        aspect_mode = job_input.get("aspect_mode", "preserve")
        if aspect_mode not in ["preserve", "crop", "stretch"]:
            return False, "aspect_mode must be one of: preserve, crop, stretch"
        
        # Validate LightX2V rank
        lightx2v_rank = job_input.get("lightx2v_rank", "32")
        if lightx2v_rank not in ["32", "64", "128"]:
            return False, "LightX2V rank must be one of: 32, 64, 128"
        
        # Validate numerical ranges
        cfg_scale = job_input.get("cfg_scale", 1.0)
        if not (0.1 <= cfg_scale <= 10.0):
            return False, "CFG scale must be between 0.1 and 10.0"
        
        steps = job_input.get("steps", 6)
        if not (1 <= steps <= 50):
            return False, "Steps must be between 1 and 50"
        
        high_noise_steps = job_input.get("high_noise_steps", 3)
        if not (1 <= high_noise_steps < steps):
            return False, f"High noise steps must be between 1 and {steps-1}"
        
        interpolation_factor = job_input.get("interpolation_factor", 2)
        if not (2 <= interpolation_factor <= 8):
            return False, "Interpolation factor must be between 2 and 8"
        
        return True, "All parameters valid"
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False, f"Validation error: {str(e)}"

# ================== MAIN HANDLER ==================

def handler(job):
    """
    🚀 COMPLETE FIXED Main RunPod handler
    ✅ Multi-size image support, comprehensive error handling, production ready
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Comprehensive validation
        is_valid, validation_message = validate_input_comprehensive(job_input)
        if not is_valid:
            return {
                "error": validation_message,
                "status": "failed",
                "job_id": job_id,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        
        # System readiness checks
        if not COMFYUI_AVAILABLE:
            return {
                "error": "ComfyUI not available due to system compatibility issues",
                "status": "failed",
                "job_id": job_id,
                "cuda_compatibility_issue": not CUDA_AVAILABLE
            }
        
        models_ok, missing_models = verify_models()
        if not models_ok:
            return {
                "error": "Required models are missing",
                "missing_models": missing_models,
                "status": "failed",
                "job_id": job_id
            }
        
        # Extract parameters
        image_url = job_input["image_url"]
        positive_prompt = job_input["positive_prompt"]
        
        logger.info(f"🚀 Job {job_id}: COMPLETE FIXED WAN2.2 generation started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"📝 Prompt: {positive_prompt[:100]}...")
        logger.info(f"⚙️ System: CUDA={CUDA_AVAILABLE}, ComfyUI={COMFYUI_AVAILABLE}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download image
            image_path = os.path.join(temp_dir, "input_image.jpg")
            
            try:
                response = requests.get(image_url, timeout=60, stream=True)
                response.raise_for_status()
                
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
                image_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                logger.info(f"✅ Image downloaded: {image_size_mb:.1f}MB")
                
            except Exception as e:
                return {"error": f"Failed to download image: {str(e)}", "status": "failed"}
            
            # Generate video với COMPLETE FIXED function
            generation_start = time.time()
            
            output_path = generate_video_wan22_complete_fixed(
                image_path=image_path,
                **job_input
            )
            
            generation_time = time.time() - generation_start
            
            if not output_path or not os.path.exists(output_path):
                return {
                    "error": "Video generation failed - no output produced",
                    "status": "failed",
                    "job_id": job_id
                }
            
            # Upload result
            output_filename = f"wan22_complete_fixed_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            
            try:
                output_url = upload_to_minio(output_path, output_filename)
            except Exception as e:
                return {
                    "error": f"Failed to upload result: {str(e)}",
                    "status": "failed",
                    "job_id": job_id
                }
            
            # Calculate final statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            duration_seconds = job_input.get("frames", 65) / job_input.get("fps", 16)
            
            # Check if interpolation was applied
            interpolated = (job_input.get("enable_interpolation", False) and 
                          "_rife_" in os.path.basename(output_path))
            
            actual_seed = job_input.get("seed", 0) if job_input.get("seed", 0) != 0 else "auto-generated"
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.1f}s (generation: {generation_time:.1f}s)")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB, {duration_seconds:.1f}s duration")
            
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(total_time, 2),
                "generation_time_seconds": round(generation_time, 2),
                
                "video_info": {
                    "width": job_input.get("width", 720),
                    "height": job_input.get("height", 1280),
                    "frames": job_input.get("frames", 65),
                    "fps": job_input.get("fps", 16),
                    "duration_seconds": round(duration_seconds, 2),
                    "file_size_mb": round(file_size_mb, 2),
                    "interpolated": interpolated,
                    "interpolation_factor": job_input.get("interpolation_factor", 1) if interpolated else 1,
                    "aspect_mode": job_input.get("aspect_mode", "preserve"),
                    "auto_720p": job_input.get("auto_720p", False),
                    "multi_size_support": True,
                    "tensor_shape_fixed": True
                },
                
                "generation_params": {
                    "positive_prompt": job_input["positive_prompt"],
                    "negative_prompt": job_input.get("negative_prompt", "")[:100] + "...",
                    "steps": job_input.get("steps", 6),
                    "high_noise_steps": job_input.get("high_noise_steps", 3),
                    "cfg_scale": job_input.get("cfg_scale", 1.0),
                    "seed": actual_seed,
                    "sampler_name": job_input.get("sampler_name", "euler"),
                    "scheduler": job_input.get("scheduler", "simple"),
                    "prompt_assist": job_input.get("prompt_assist", "none"),
                    
                    "lightx2v_config": {
                        "enabled": job_input.get("use_lightx2v", True),
                        "rank": job_input.get("lightx2v_rank", "32"),
                        "strength": job_input.get("lightx2v_strength", 3.0)
                    },
                    
                    "pusa_config": {
                        "enabled": job_input.get("use_pusa", True),
                        "strength": job_input.get("pusa_strength", 1.5)
                    },
                    
                    "lora_config": {
                        "enabled": job_input.get("use_lora", False),
                        "url": job_input.get("lora_url"),
                        "strength": job_input.get("lora_strength", 1.0)
                    },
                    
                    "aspect_control": {
                        "mode": job_input.get("aspect_mode", "preserve"),
                        "auto_720p": job_input.get("auto_720p", False)
                    },
                    
                    "interpolation": {
                        "enabled": job_input.get("enable_interpolation", False),
                        "factor": job_input.get("interpolation_factor", 2)
                    },
                    
                    "optimizations": {
                        "sage_attention": job_input.get("use_sage_attention", True),
                        "teacache_threshold": job_input.get("rel_l1_thresh", 0.0)
                    },
                    
                    "system_info": {
                        "cuda_available": CUDA_AVAILABLE,
                        "comfyui_available": COMFYUI_AVAILABLE,
                        "model_quantization": "Q6_K",
                        "workflow_version": "COMPLETE_FIXED_v4.0",
                        "multi_size_support": True,
                        "tensor_shape_fixed": True
                    }
                },
                
                "status": "completed"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": f"System error: {error_msg}",
            "status": "failed",
            "job_id": job_id,
            "processing_time_seconds": round(time.time() - start_time, 2),
            "system_info": {
                "cuda_available": CUDA_AVAILABLE,
                "comfyui_available": COMFYUI_AVAILABLE,
                "error_type": type(e).__name__,
                "tensor_shape_issue": "tensor" in error_msg.lower() or "shape" in error_msg.lower()
            }
        }
    finally:
        clear_memory_enhanced()

# ================== HEALTH CHECK ==================

def health_check():
    """🏥 Comprehensive system health check"""
    try:
        issues = []
        
        # CUDA check
        if not torch.cuda.is_available():
            issues.append("CUDA not available")
        elif not CUDA_AVAILABLE:
            issues.append("CUDA compatibility issues")
        
        # Memory check
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_gb < 8:
                issues.append(f"Low GPU memory: {memory_gb:.1f}GB")
        
        # ComfyUI check
        if not COMFYUI_AVAILABLE:
            issues.append("ComfyUI not available")
        
        # Models check
        models_ok, missing = verify_models()
        if not models_ok:
            issues.append(f"Missing {len(missing)} models")
        
        # Storage check
        if not minio_client:
            issues.append("Storage not available")
        
        # Directory checks
        required_dirs = ["/app/ComfyUI", "/app/Practical-RIFE"]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                issues.append(f"Missing directory: {dir_path}")
        
        if issues:
            return False, f"Health issues: {'; '.join(issues)}"
        else:
            return True, "All systems operational - COMPLETE FIXED READY"
            
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

# ================== MAIN ENTRY POINT ==================

if __name__ == "__main__":
    logger.info("🚀 Starting WAN2.2 COMPLETE FIXED Serverless Worker...")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"🛡️ CUDA Safety: {CUDA_AVAILABLE}")
    logger.info(f"📦 ComfyUI Available: {COMFYUI_AVAILABLE}")
    
    if torch.cuda.is_available():
        try:
            logger.info(f"💾 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        except:
            logger.warning("⚠️ Could not get GPU details")
    
    try:
        # Comprehensive health check
        health_ok, health_msg = health_check()
        
        logger.info(f"📊 System Status: {health_msg}")
        
        if not health_ok:
            logger.warning("⚠️ System has health issues but attempting to continue...")
        
        # Feature summary
        logger.info("📋 COMPLETE FIXED Features:")
        logger.info("  ✅ Multi-Size Image Support (256-1536px)")
        logger.info("  ✅ Tensor Shape Fixes for All Common Sizes")
        logger.info("  ✅ CUDA Compatibility Fixes")
        logger.info("  ✅ Enhanced Aspect Ratio Control")
        logger.info("  ✅ RIFE Frame Interpolation (Ultimate Fixed)")
        logger.info("  ✅ LightX2V LoRA Support (32/64/128)")
        logger.info("  ✅ Custom LoRA Support (HF/CivitAI)")
        logger.info("  ✅ Advanced Optimizations")
        logger.info("  ✅ Production-Grade Error Handling")
        logger.info("  ✅ Comprehensive Validation & Monitoring")
        
        logger.info("🎬 COMPLETE FIXED READY - Optimized for all image sizes!")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
