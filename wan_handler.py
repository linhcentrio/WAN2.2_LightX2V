#!/usr/bin/env python3
"""
🚀 RunPod Serverless Handler cho WAN2.2 LightX2V Q6_K - PRODUCTION FINAL VERSION
🔧 All Fixes Applied: CUDA Compatibility, Tensor Shapes, Aspect Ratios, RIFE Interpolation
✨ Production Ready với comprehensive error handling và optimization
📋 Supports multiple image sizes và aspect ratios một cách chính xác
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
# 🛡️ Comprehensive CUDA compatibility fixes
os.environ.update({
    'CUDA_LAUNCH_BLOCKING': '1',
    'TORCH_USE_CUDA_DSA': '1',
    'CUDA_VISIBLE_DEVICES': '0',
    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True',
    'PYTORCH_KERNEL_CACHE_PATH': '/tmp/pytorch_kernel_cache',
    'CUDA_MODULE_LOADING': 'LAZY',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'PYTORCH_CUDA_ALLOC_SYNC_MEMORY': '1',
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'
})

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add system paths
sys.path.insert(0, '/app/ComfyUI')
sys.path.insert(0, '/app/Practical-RIFE')

# ================== CUDA-SAFE PYTORCH CONFIGURATION ==================
def setup_pytorch_production():
    """🔧 Production-grade PyTorch setup với comprehensive compatibility"""
    try:
        if torch.cuda.is_available():
            # 🛡️ Conservative CUDA settings cho maximum compatibility
            torch.backends.cudnn.benchmark = False  # Disable auto-tuning
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True  # Ensure reproducibility
            torch.set_float32_matmul_precision('highest')  # Most stable precision
            
            # Disable problematic optimizations
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            
            # Test CUDA functionality
            try:
                device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device)
                compute_cap = torch.cuda.get_device_capability(device)
                memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
                
                logger.info(f"🎯 GPU: {device_name}")
                logger.info(f"🔧 Compute: {compute_cap}, Memory: {memory_gb:.1f}GB")
                
                # Test basic operations
                test_tensor = torch.ones(10, device='cuda', dtype=torch.float32)
                test_result = test_tensor.sum()
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
                # Test embedding operations (critical for text encoder)
                test_embed = torch.nn.Embedding(50, 16, dtype=torch.float32).cuda()
                test_input = torch.randint(0, 50, (2, 5), device='cuda')
                test_output = test_embed(test_input)
                del test_embed, test_input, test_output
                torch.cuda.empty_cache()
                
                logger.info("✅ CUDA compatibility verification passed")
                return True
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['kernel', 'no kernel', 'compatibility']):
                    logger.error(f"❌ CUDA kernel compatibility issue: {e}")
                    # Apply additional compatibility measures
                    torch.backends.cudnn.enabled = False
                    logger.info("🔄 Applied CUDNN disable fix")
                    return True  # Continue with CUDNN disabled
                else:
                    raise e
        else:
            logger.warning("⚠️ CUDA not available, running in CPU mode")
            return False
            
    except Exception as e:
        logger.error(f"❌ PyTorch setup failed: {e}")
        return False

# Initialize CUDA safety
CUDA_AVAILABLE = setup_pytorch_production()

# ================== SAFE COMFYUI IMPORTS ==================
def safe_import_comfyui():
    """🔧 Production-safe ComfyUI import với comprehensive error handling"""
    try:
        logger.info("📦 Importing ComfyUI modules...")
        
        # Core ComfyUI nodes
        from nodes import (
            CheckpointLoaderSimple, CLIPLoader, CLIPTextEncode, VAEDecode, VAELoader,
            KSampler, KSamplerAdvanced, UNETLoader, LoadImage, SaveImage,
            CLIPVisionLoader, CLIPVisionEncode, LoraLoaderModelOnly, ImageScale
        )
        
        # GGUF support
        from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
        
        # Optimization nodes
        from custom_nodes.ComfyUI_KJNodes.nodes.model_optimization_nodes import (
            WanVideoTeaCacheKJ, PathchSageAttentionKJ, WanVideoNAG, SkipLayerGuidanceWanVideo
        )
        
        # Advanced nodes
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes_images import SaveAnimatedWEBP
        from comfy_extras.nodes_video import SaveWEBM
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
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, {}

# Import ComfyUI components
COMFYUI_AVAILABLE, COMFYUI_NODES = safe_import_comfyui()

# ================== MODEL & STORAGE CONFIGURATIONS ==================
MODEL_CONFIGS = {
    # Q6_K DIT Models
    "dit_model_high": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
    "dit_model_low": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf",
    
    # Supporting models
    "text_encoder": "/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "/app/ComfyUI/models/vae/wan_2.1_vae.safetensors",
    "clip_vision": "/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
    
    # LightX2V LoRAs
    "lightx2v_rank_32": "/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
    "lightx2v_rank_64": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
    "lightx2v_rank_128": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
    
    # Built-in LoRAs
    "walking_to_viewers": "/app/ComfyUI/models/loras/walking to viewers_Wan.safetensors",
    "walking_from_behind": "/app/ComfyUI/models/loras/walking_from_behind.safetensors",
    "dancing": "/app/ComfyUI/models/loras/b3ll13-d8nc3r.safetensors",
    "pusa_lora": "/app/ComfyUI/models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors",
    "rotate_lora": "/app/ComfyUI/models/loras/rotate_20_epochs.safetensors"
}

# MinIO Storage Configuration
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

def clear_memory_comprehensive():
    """🧹 Comprehensive memory cleanup với enhanced CUDA management"""
    try:
        # Python garbage collection
        collected = gc.collect()
        
        if CUDA_AVAILABLE and torch.cuda.is_available():
            # Multi-stage CUDA cleanup
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
            # Reset memory statistics
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Force memory cleanup
            if hasattr(torch.cuda, 'memory_summary'):
                try:
                    torch.cuda.memory_summary(device=None, abbreviated=True)
                except:
                    pass
                    
            logger.debug(f"🧹 Memory cleanup: collected {collected} objects")
        
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup warning: {e}")

def verify_models_comprehensive() -> tuple[bool, list]:
    """🔍 Comprehensive model verification với detailed reporting"""
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
                
                # Verify file integrity (basic check)
                if file_size_mb < 1:  # Suspiciously small
                    logger.warning(f"⚠️ {name} seems too small: {file_size_mb:.1f}MB")
                    
            except Exception as e:
                logger.error(f"❌ Error checking {name}: {e}")
                missing_models.append(f"{name}: {path} (read error)")
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
    """📍 Get LightX2V LoRA path by rank với validation"""
    rank_mapping = {
        "32": "lightx2v_rank_32",
        "64": "lightx2v_rank_64",
        "128": "lightx2v_rank_128"
    }
    config_key = rank_mapping.get(lightx2v_rank, "lightx2v_rank_32")
    path = MODEL_CONFIGS.get(config_key)
    
    if path and os.path.exists(path):
        return path
    else:
        logger.warning(f"⚠️ LightX2V rank {lightx2v_rank} not found, using default")
        return MODEL_CONFIGS.get("lightx2v_rank_32")

def download_lora_dynamic(lora_url: str, civitai_token: str = None) -> str:
    """🎨 Enhanced LoRA download với comprehensive error handling"""
    try:
        lora_dir = "/app/ComfyUI/models/loras"
        os.makedirs(lora_dir, exist_ok=True)
        
        logger.info(f"🎨 Downloading LoRA: {lora_url}")
        
        if "huggingface.co" in lora_url:
            # HuggingFace download
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
                
                # Verify download
                if os.path.exists(downloaded_path) and os.path.getsize(downloaded_path) > 0:
                    size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
                    logger.info(f"✅ HuggingFace LoRA downloaded: {filename} ({size_mb:.1f}MB)")
                    return downloaded_path
                else:
                    logger.error("❌ HuggingFace download verification failed")
                    return None
                
        elif "civitai.com" in lora_url:
            # CivitAI download với enhanced handling
            try:
                if "/models/" in lora_url:
                    model_id = lora_url.split("/models/")[1].split("?")[0].split("/")[0]
                else:
                    raise ValueError("Invalid CivitAI URL format")
                
                headers = {}
                if civitai_token:
                    headers["Authorization"] = f"Bearer {civitai_token}"
                
                api_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
                logger.info(f"📥 CivitAI: model_id={model_id}")
                
                response = requests.get(api_url, headers=headers, timeout=300, stream=True)
                response.raise_for_status()
                
                timestamp = int(time.time())
                filename = f"civitai_{model_id}_{timestamp}.safetensors"
                local_path = os.path.join(lora_dir, filename)
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress logging every 10MB
                            if total_size > 0 and downloaded % (1024 * 1024 * 10) == 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"📥 Download progress: {progress:.1f}%")
                
                # Verify download
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    size_mb = downloaded / (1024 * 1024)
                    logger.info(f"✅ CivitAI LoRA downloaded: {filename} ({size_mb:.1f}MB)")
                    return local_path
                else:
                    logger.error("❌ CivitAI download verification failed")
                    return None
                
            except Exception as e:
                logger.error(f"❌ CivitAI download failed: {e}")
                return None
        else:
            # Direct download
            filename = os.path.basename(urlparse(lora_url).path)
            if not filename.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                filename = f"direct_lora_{int(time.time())}.safetensors"
            
            local_path = os.path.join(lora_dir, filename)
            
            response = requests.get(lora_url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify download
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                logger.info(f"✅ Direct LoRA downloaded: {filename} ({size_mb:.1f}MB)")
                return local_path
            else:
                logger.error("❌ Direct download verification failed")
                return None
        
        return None
        
    except Exception as e:
        logger.error(f"❌ LoRA download failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# ================== ENHANCED ASPECT RATIO PROCESSING ==================

def calculate_aspect_preserving_dimensions(original_width, original_height, target_width, target_height):
    """📐 Calculate dimensions preserving aspect ratio với enhanced logic"""
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height
    
    if abs(original_ratio - target_ratio) < 0.01:  # Already close to target ratio
        return target_width, target_height
    
    if original_ratio > target_ratio:
        # Wider image -> pad top/bottom
        new_width = target_width
        new_height = int(target_width / original_ratio)
    else:
        # Taller image -> pad left/right
        new_height = target_height
        new_width = int(target_height * original_ratio)
    
    # Ensure divisible by 8 for video encoding
    new_width = max(8, (new_width // 8) * 8)
    new_height = max(8, (new_height // 8) * 8)
    
    return new_width, new_height

def calculate_crop_dimensions(original_width, original_height, target_width, target_height):
    """✂️ Calculate smart crop dimensions với center focus"""
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    
    if original_ratio > target_ratio:
        # Wider image -> crop width (center crop)
        crop_height = original_height
        crop_width = int(original_height * target_ratio)
        crop_x = max(0, (original_width - crop_width) // 2)
        crop_y = 0
    else:
        # Taller image -> crop height (center crop)
        crop_width = original_width
        crop_height = int(original_width / target_ratio)
        crop_x = 0
        crop_y = max(0, (original_height - crop_height) // 2)
    
    # Ensure crop dimensions are valid
    crop_width = min(crop_width, original_width)
    crop_height = min(crop_height, original_height)
    crop_x = min(crop_x, original_width - crop_width)
    crop_y = min(crop_y, original_height - crop_height)
    
    return crop_x, crop_y, crop_width, crop_height

def process_image_with_enhanced_aspect_control(
    image_path: str,
    target_width: int = 720,
    target_height: int = 1280,
    aspect_mode: str = "preserve",
    auto_720p: bool = False
):
    """
    🖼️ PRODUCTION: Enhanced image processing với comprehensive tensor validation
    Supports all popular image sizes và aspect ratios
    """
    try:
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI not available")
        
        load_image = COMFYUI_NODES['LoadImage']()
        image_scaler = COMFYUI_NODES['ImageScale']()
        
        # Load và validate original image
        loaded_image = load_image.load_image(image_path)[0]
        
        logger.info(f"📊 Initial image tensor: shape={loaded_image.shape}, dtype={loaded_image.dtype}")
        
        # Normalize tensor dimensions
        if loaded_image.ndim == 4:
            batch, orig_height, orig_width, channels = loaded_image.shape
            if batch != 1:
                logger.warning(f"⚠️ Unexpected batch size: {batch}")
                loaded_image = loaded_image[0:1]
        elif loaded_image.ndim == 3:
            orig_height, orig_width, channels = loaded_image.shape
            loaded_image = loaded_image.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Invalid image tensor shape: {loaded_image.shape}")
        
        logger.info(f"🖼️ Original image: {orig_width}x{orig_height} (channels: {channels})")
        
        # Auto 720p detection với enhanced logic
        if auto_720p:
            aspect_ratio = orig_width / orig_height
            
            if aspect_ratio > 1.3:  # Landscape
                target_width, target_height = 1280, 720
            elif aspect_ratio < 0.8:  # Portrait
                target_width, target_height = 720, 1280
            else:  # Square-ish, choose based on original size
                if orig_width > orig_height:
                    target_width, target_height = 1280, 720
                else:
                    target_width, target_height = 720, 1280
                    
            logger.info(f"📱 Auto 720p: {orig_width}x{orig_height} -> {target_width}x{target_height}")
        
        # Process based on aspect mode với comprehensive validation
        if aspect_mode == "preserve":
            # Calculate new dimensions preserving aspect ratio
            new_width, new_height = calculate_aspect_preserving_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            logger.info(f"📐 Preserve dimensions: {new_width}x{new_height}")
            
            # Scale to calculated dimensions
            scaled_image = image_scaler.upscale(
                loaded_image, "lanczos", new_width, new_height, "disabled"
            )[0]
            
            logger.info(f"📊 Scaled tensor: {scaled_image.shape}")
            
            # Add padding if needed
            if new_width != target_width or new_height != target_height:
                logger.info(f"🔧 Adding padding: {new_width}x{new_height} -> {target_width}x{target_height}")
                
                # Calculate symmetric padding
                pad_x = (target_width - new_width) // 2
                pad_y = (target_height - new_height) // 2
                pad_x_right = target_width - new_width - pad_x
                pad_y_bottom = target_height - new_height - pad_y
                
                logger.info(f"📏 Padding: left={pad_x}, right={pad_x_right}, top={pad_y}, bottom={pad_y_bottom}")
                
                # Ensure proper tensor format
                if scaled_image.ndim == 3:
                    scaled_image = scaled_image.unsqueeze(0)
                
                # Convert BHWC -> BCHW for padding
                scaled_bchw = scaled_image.permute(0, 3, 1, 2)
                
                # Apply padding với validation
                try:
                    padded_bchw = F.pad(
                        scaled_bchw,
                        (pad_x, pad_x_right, pad_y, pad_y_bottom),
                        mode='constant',
                        value=0
                    )
                    
                    # Convert back BCHW -> HWC
                    final_image = padded_bchw.permute(0, 2, 3, 1)[0]
                    
                except Exception as pad_error:
                    logger.error(f"❌ Padding failed: {pad_error}")
                    logger.info("🔄 Fallback to direct scaling")
                    final_image = image_scaler.upscale(
                        loaded_image, "lanczos", target_width, target_height, "disabled"
                    )[0]
            else:
                final_image = scaled_image if scaled_image.ndim == 3 else scaled_image[0]
                
        elif aspect_mode == "crop":
            # Smart crop preserving aspect ratio
            crop_x, crop_y, crop_width, crop_height = calculate_crop_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            logger.info(f"✂️ Crop: ({crop_x},{crop_y}) {crop_width}x{crop_height}")
            
            # Perform crop với bounds checking
            if loaded_image.ndim == 4:
                cropped = loaded_image[0, crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
                cropped = cropped.unsqueeze(0)
            else:
                cropped = loaded_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
                cropped = cropped.unsqueeze(0)
            
            # Scale to target size
            final_image = image_scaler.upscale(
                cropped, "lanczos", target_width, target_height, "disabled"
            )[0]
            
        else:  # stretch
            # Direct stretch (backward compatible)
            final_image = image_scaler.upscale(
                loaded_image, "lanczos", target_width, target_height, "disabled"
            )[0]
        
        # COMPREHENSIVE FINAL VALIDATION
        logger.info(f"🔍 Final image validation:")
        logger.info(f"  Shape: {final_image.shape}")
        logger.info(f"  Expected: [{target_height}, {target_width}, 3]")
        
        # Validate tensor properties
        if final_image.ndim != 3:
            raise ValueError(f"Final image must be 3D [H,W,C], got {final_image.ndim}D")
        
        actual_h, actual_w, actual_c = final_image.shape
        
        # Dimension validation
        if actual_h != target_height or actual_w != target_width:
            logger.error(f"❌ Size mismatch: {actual_h}x{actual_w} vs {target_height}x{target_width}")
            raise ValueError(f"Size mismatch: expected {target_height}x{target_width}, got {actual_h}x{actual_w}")
        
        # Channel validation và normalization
        if actual_c == 1:
            logger.info("🔄 Converting grayscale to RGB...")
            final_image = final_image.repeat(1, 1, 3)
        elif actual_c == 4:
            logger.info("🔄 Removing alpha channel...")
            final_image = final_image[:, :, :3]
        elif actual_c != 3:
            raise ValueError(f"Invalid channel count: {actual_c}")
        
        # Final validation
        final_h, final_w, final_c = final_image.shape
        if final_c != 3:
            raise ValueError(f"Final image must have 3 channels, got {final_c}")
        
        logger.info(f"✅ Image processing completed: {final_w}x{final_h} RGB")
        logger.info(f"📊 Mode: {aspect_mode}, Auto720p: {auto_720p}")
        
        return final_image, target_width, target_height
        
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

# ================== CUDA-SAFE TEXT PROCESSING ==================

def safe_load_text_encoder():
    """🔧 Production-safe text encoder loading với comprehensive fallbacks"""
    try:
        clear_memory_comprehensive()
        
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI not available")
        
        clip_loader = COMFYUI_NODES['CLIPLoader']()
        
        # Strategy 1: Normal loading
        try:
            logger.info("📝 Loading text encoder...")
            clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            
            # Test functionality
            clip_encode = COMFYUI_NODES['CLIPTextEncode']()
            test_result = clip_encode.encode(clip, "test")
            del test_result
            
            logger.info("✅ Text encoder loaded successfully")
            return clip
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['cuda error', 'kernel', 'no kernel']):
                logger.warning("⚠️ CUDA compatibility issue, applying fixes...")
                
                # Strategy 2: Disable CUDNN
                original_cudnn = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                
                try:
                    clear_memory_comprehensive()
                    clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
                    logger.info("✅ Text encoder loaded with CUDNN disabled")
                    return clip
                except Exception:
                    torch.backends.cudnn.enabled = original_cudnn
                
                # Strategy 3: Different device context
                try:
                    clear_memory_comprehensive()
                    with torch.cuda.device(0):
                        clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
                    logger.info("✅ Text encoder loaded with device context")
                    return clip
                except Exception:
                    pass
                
                # Strategy 4: CPU fallback (last resort)
                logger.warning("🔄 CPU fallback for text encoder...")
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
    """🔧 CUDA-safe text encoding với comprehensive retry logic"""
    try:
        clip_encode = COMFYUI_NODES['CLIPTextEncode']()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = clip_encode.encode(clip, text)
                return result[0]
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['cuda error', 'kernel']) and attempt < max_retries - 1:
                    logger.warning(f"⚠️ CUDA error on attempt {attempt + 1}, retrying...")
                    clear_memory_comprehensive()
                    time.sleep(1)  # Brief pause
                    continue
                else:
                    raise e
                    
    except Exception as e:
        logger.error(f"❌ Text encoding failed: {e}")
        raise e

# ================== RIFE INTERPOLATION (PRODUCTION FIXED) ==================

def apply_rife_interpolation_production_final(video_path: str, interpolation_factor: int = 2) -> str:
    """🔧 PRODUCTION FINAL: RIFE interpolation với ultimate output detection"""
    try:
        logger.info(f"🔄 RIFE interpolation (factor: {interpolation_factor}) - PRODUCTION FINAL...")
        
        # Validate input
        if not os.path.exists(video_path):
            logger.error(f"❌ Video not found: {video_path}")
            return video_path
        
        original_size = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"📊 Original: {original_size:.1f}MB")
        
        # Check RIFE availability
        rife_script = "/app/Practical-RIFE/inference_video.py"
        if not os.path.exists(rife_script):
            logger.warning(f"⚠️ RIFE not available: {rife_script}")
            return video_path
        
        # Setup directories
        rife_output_dir = "/app/rife_output"
        os.makedirs(rife_output_dir, exist_ok=True)
        
        # Generate output path
        input_basename = os.path.splitext(os.path.basename(video_path))[0]
        final_output_path = os.path.join(rife_output_dir, f"{input_basename}_rife_{interpolation_factor}x.mp4")
        
        # Copy input to RIFE directory
        rife_input_path = os.path.join("/app/Practical-RIFE", os.path.basename(video_path))
        shutil.copy2(video_path, rife_input_path)
        
        # Enhanced environment
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": "/app/Practical-RIFE",
            "SDL_AUDIODRIVER": "dummy",
            "FFMPEG_LOGLEVEL": "quiet"
        })
        
        # Build RIFE command
        cmd = [
            "python3", "inference_video.py",
            f"--multi={interpolation_factor}",
            f"--video={os.path.basename(video_path)}",
            "--scale=1.0",
            "--fps=30"
        ]
        
        logger.info(f"🔧 RIFE command: {' '.join(cmd)}")
        
        # Execute RIFE
        original_cwd = os.getcwd()
        os.chdir("/app/Practical-RIFE")
        
        try:
            # Clean previous outputs
            for pattern in ["*_interpolated.mp4", "*X.mp4", "*_rife.mp4"]:
                for file in Path(".").glob(pattern):
                    try:
                        file.unlink()
                    except:
                        pass
            
            # Run RIFE với timeout
            start_time = time.time()
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True,
                timeout=600, check=False  # 10 minutes timeout
            )
            
            execution_time = time.time() - start_time
            logger.info(f"⏱️ RIFE execution: {execution_time:.1f}s, code: {result.returncode}")
            
        except subprocess.TimeoutExpired:
            logger.error("❌ RIFE timed out")
            return video_path
        except Exception as e:
            logger.error(f"❌ RIFE execution failed: {e}")
            return video_path
        finally:
            os.chdir(original_cwd)
            # Cleanup input copy
            try:
                if os.path.exists(rife_input_path):
                    os.remove(rife_input_path)
            except:
                pass
        
        # ULTIMATE OUTPUT DETECTION
        logger.info("🔍 Ultimate output detection...")
        
        potential_outputs = []
        rife_dir = "/app/Practical-RIFE"
        input_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Comprehensive search patterns
        patterns = [
            f"{input_name}_{interpolation_factor}X.mp4",
            f"{input_name}_{interpolation_factor}x.mp4",
            f"{input_name}_interpolated.mp4",
            f"{input_name}_rife.mp4",
            f"output_{interpolation_factor}x.mp4",
            "output.mp4", "interpolated.mp4",
            os.path.basename(video_path).replace('.mp4', f'_{interpolation_factor}X.mp4'),
            os.path.basename(video_path).replace('.mp4', '_interpolated.mp4')
        ]
        
        # Search by patterns
        for pattern in patterns:
            path = os.path.join(rife_dir, pattern)
            if os.path.exists(path) and path not in potential_outputs:
                potential_outputs.append(path)
                logger.info(f"🔍 Found: {pattern}")
        
        # Search by recent modification
        current_time = time.time()
        try:
            for file in os.listdir(rife_dir):
                if file.endswith('.mp4') and file != os.path.basename(video_path):
                    file_path = os.path.join(rife_dir, file)
                    # Check if created in last 15 minutes
                    if current_time - max(os.path.getctime(file_path), os.path.getmtime(file_path)) < 900:
                        if file_path not in potential_outputs:
                            potential_outputs.append(file_path)
                            logger.info(f"🔍 Recent: {file}")
        except Exception as e:
            logger.warning(f"⚠️ Directory scan failed: {e}")
        
        # Select best output
        selected_output = None
        if potential_outputs:
            logger.info(f"📊 Analyzing {len(potential_outputs)} candidates:")
            
            best_candidate = None
            best_score = 0
            
            for path in potential_outputs:
                try:
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    mod_time = os.path.getmtime(path)
                    
                    # Scoring: size relevance + recency
                    size_score = min(size_mb / original_size, 5.0)
                    recency_score = max(0, (900 - (current_time - mod_time)) / 900)
                    total_score = size_score * 0.8 + recency_score * 0.2
                    
                    logger.info(f"  📄 {os.path.basename(path)}: {size_mb:.1f}MB, score: {total_score:.2f}")
                    
                    if (size_mb >= original_size * 0.3 and  # Minimum 30% of original
                        size_mb <= original_size * 20 and    # Maximum 20x original
                        total_score > best_score):
                        
                        best_candidate = path
                        best_score = total_score
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error analyzing {path}: {e}")
            
            selected_output = best_candidate
        
        # Process selected output
        if selected_output and os.path.exists(selected_output):
            try:
                shutil.move(selected_output, final_output_path)
                
                final_size = os.path.getsize(final_output_path) / (1024 * 1024)
                logger.info(f"✅ RIFE interpolation successful!")
                logger.info(f"📊 {original_size:.1f}MB → {final_size:.1f}MB ({final_size/original_size:.1f}x)")
                logger.info(f"⏱️ Processing time: {execution_time:.1f}s")
                
                return final_output_path
                
            except Exception as e:
                logger.error(f"❌ Failed to move output: {e}")
                return video_path
        else:
            # Debug logging
            logger.warning("🔍 COMPREHENSIVE DEBUG - No valid output found")
            try:
                logger.warning("📁 RIFE directory contents:")
                for item in os.listdir(rife_dir):
                    if os.path.isfile(os.path.join(rife_dir, item)):
                        size = os.path.getsize(os.path.join(rife_dir, item)) / (1024 * 1024)
                        logger.warning(f"  📄 {item}: {size:.1f}MB")
            except Exception as e:
                logger.error(f"❌ Debug listing failed: {e}")
            
            logger.warning("⚠️ Using original video")
            return video_path
            
    except Exception as e:
        logger.error(f"❌ RIFE interpolation error: {e}")
        return video_path

# ================== PRODUCTION VIDEO SAVING ==================

def save_video_production_final(frames_tensor, output_path, fps=16):
    """🎬 PRODUCTION FINAL: Multi-strategy video saving với comprehensive validation"""
    try:
        logger.info(f"🎬 Saving video: {output_path}")
        
        if frames_tensor is None:
            raise ValueError("Frames tensor is None")
        
        # Convert to numpy với validation
        if torch.is_tensor(frames_tensor):
            frames_np = frames_tensor.detach().cpu().float().numpy()
        else:
            frames_np = np.array(frames_tensor, dtype=np.float32)
        
        logger.info(f"📊 Input: shape={frames_np.shape}, dtype={frames_np.dtype}, range=[{frames_np.min():.3f}, {frames_np.max():.3f}]")
        
        # Handle batch dimension
        if frames_np.ndim == 5 and frames_np.shape[0] == 1:
            frames_np = frames_np[0]
            logger.info(f"📊 Batch removed: {frames_np.shape}")
        
        # Convert to uint8
        if frames_np.dtype != np.uint8:
            if frames_np.max() <= 1.0:
                frames_np = (frames_np * 255.0).astype(np.uint8)
            else:
                frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)
        
        # Validate dimensions
        if len(frames_np.shape) != 4:
            raise ValueError(f"Expected 4D tensor [frames, height, width, channels], got {frames_np.shape}")
        
        num_frames, height, width, channels = frames_np.shape
        logger.info(f"📊 Video specs: {num_frames} frames, {width}x{height}, {channels} channels")
        
        if num_frames == 0:
            raise ValueError("No frames to save")
        
        # Handle different channel configurations
        if channels == 1:
            logger.info("🔄 Converting grayscale to RGB...")
            frames_np = np.repeat(frames_np, 3, axis=-1)
            channels = 3
        elif channels == 4:
            logger.info("🔄 Removing alpha channel...")
            frames_np = frames_np[:, :, :, :3]
            channels = 3
        elif channels != 3:
            raise ValueError(f"Unsupported channel count: {channels}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Multi-strategy encoding
        strategies = [
            {
                'name': 'High Quality',
                'params': {
                    'fps': fps,
                    'codec': 'libx264',
                    'pixelformat': 'yuv420p',
                    'quality': 8,
                    'bitrate': '5M'
                }
            },
            {
                'name': 'Standard',
                'params': {
                    'fps': fps,
                    'codec': 'libx264',
                    'pixelformat': 'yuv420p'
                }
            },
            {
                'name': 'Basic',
                'params': {'fps': fps}
            },
            {
                'name': 'Fallback',
                'params': {
                    'fps': fps,
                    'codec': 'libx264',
                    'macro_block_size': None
                }
            }
        ]
        
        # Try encoding strategies
        for strategy in strategies:
            try:
                logger.info(f"🎥 Trying {strategy['name']} encoding...")
                
                with imageio.get_writer(output_path, **strategy['params']) as writer:
                    for i, frame in enumerate(frames_np):
                        writer.append_data(frame)
                        
                        # Progress logging
                        if (i + 1) % max(1, num_frames // 5) == 0:
                            progress = ((i + 1) / num_frames) * 100
                            logger.info(f"📹 Progress: {progress:.1f}% ({i + 1}/{num_frames})")
                
                # Verify output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    # Basic sanity check
                    if file_size_mb < 0.01:  # Less than 10KB is suspicious
                        logger.warning(f"⚠️ {strategy['name']} produced suspiciously small file: {file_size_mb:.3f}MB")
                        continue
                    
                    logger.info(f"✅ Video saved with {strategy['name']}!")
                    logger.info(f"📊 Final: {file_size_mb:.2f}MB, {num_frames} frames @ {fps}fps")
                    
                    return output_path
                else:
                    logger.warning(f"⚠️ {strategy['name']} produced empty file")
                    
            except Exception as e:
                logger.warning(f"⚠️ {strategy['name']} failed: {e}")
                continue
        
        raise RuntimeError("All encoding strategies failed")
        
    except Exception as e:
        logger.error(f"❌ Video saving failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

# ================== MAIN VIDEO GENERATION (PRODUCTION FINAL) ==================

def generate_video_wan22_production_final(image_path: str, **kwargs) -> str:
    """
    🎬 PRODUCTION FINAL: WAN2.2 video generation với comprehensive optimization
    Supports all popular image sizes và aspect ratios
    """
    try:
        logger.info("🎬 Starting WAN2.2 PRODUCTION FINAL generation...")
        
        # Parameter extraction với comprehensive defaults
        positive_prompt = kwargs.get('positive_prompt', '')
        negative_prompt = kwargs.get('negative_prompt', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
        
        # Enhanced aspect control
        aspect_mode = kwargs.get('aspect_mode', 'preserve')
        auto_720p = kwargs.get('auto_720p', False)
        
        # Video settings
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
        
        # Optimization settings
        use_sage_attention = kwargs.get('use_sage_attention', True)
        rel_l1_thresh = kwargs.get('rel_l1_thresh', 0.0)
        
        # Frame interpolation
        enable_interpolation = kwargs.get('enable_interpolation', False)
        interpolation_factor = kwargs.get('interpolation_factor', 2)
        
        # Generate seed
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
        
        logger.info(f"🎯 Configuration Summary:")
        logger.info(f"  📐 Resolution: {width}x{height} (mode: {aspect_mode}, auto720p: {auto_720p})")
        logger.info(f"  🎬 Animation: {frames} frames @ {fps}fps")
        logger.info(f"  ⚙️ Sampling: {steps} steps (high: {high_noise_steps}), CFG: {cfg_scale}")
        logger.info(f"  🌱 Seed: {seed}, Sampler: {sampler_name}")
        logger.info(f"  🎭 Features: assist={prompt_assist}, lightx2v={lightx2v_rank}, pusa={use_pusa}")
        logger.info(f"  🔄 Interpolation: {enable_interpolation} (factor: {interpolation_factor})")
        
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI not available")
        
        with torch.inference_mode():
            # Initialize nodes
            logger.info("🔧 Initializing ComfyUI nodes...")
            unet_loader = COMFYUI_NODES['UnetLoaderGGUF']()
            pathch_sage_attention = COMFYUI_NODES['PathchSageAttentionKJ']()
            wan_video_nag = COMFYUI_NODES['WanVideoNAG']()
            teacache = COMFYUI_NODES['WanVideoTeaCacheKJ']()
            model_sampling = COMFYUI_NODES['ModelSamplingSD3']()
            vae_loader = COMFYUI_NODES['VAELoader']()
            wan_image_to_video = COMFYUI_NODES['WanImageToVideo']()
            ksampler = COMFYUI_NODES['KSamplerAdvanced']()
            vae_decode = COMFYUI_NODES['VAEDecode']()
            
            # LoRA loaders
            prompt_assist_lora = COMFYUI_NODES['LoraLoaderModelOnly']()
            custom_lora = COMFYUI_NODES['LoraLoaderModelOnly']()
            lightx2v_lora = COMFYUI_NODES['LoraLoaderModelOnly']()
            pusa_lora = COMFYUI_NODES['LoraLoaderModelOnly']()
            
            logger.info("✅ All nodes initialized")
            
            # Text encoding với CUDA safety
            logger.info("📝 Text encoding...")
            clip = safe_load_text_encoder()
            
            # Process prompts
            final_positive_prompt = positive_prompt
            if prompt_assist != "none":
                final_positive_prompt = f"{positive_prompt} {prompt_assist}."
                logger.info(f"🎭 Enhanced prompt: {final_positive_prompt[:80]}...")
            
            positive = safe_encode_text(clip, final_positive_prompt)
            negative = safe_encode_text(clip, negative_prompt)
            
            del clip
            clear_memory_comprehensive()
            
            # Image processing với enhanced aspect control
            logger.info("🖼️ Enhanced image processing...")
            loaded_image, final_width, final_height = process_image_with_enhanced_aspect_control(
                image_path=image_path,
                target_width=width,
                target_height=height,
                aspect_mode=aspect_mode,
                auto_720p=auto_720p
            )
            
            # Update dimensions
            width, height = final_width, final_height
            
            # VAE loading
            logger.info("🎨 Loading VAE...")
            vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            
            # Critical: Image to video encoding với comprehensive validation
            logger.info("🔄 Image to video encoding...")
            logger.info(f"🔍 Pre-encode validation:")
            logger.info(f"  Image: {loaded_image.shape} {loaded_image.dtype}")
            logger.info(f"  Target: {width}x{height}")
            logger.info(f"  Frames: {frames}")
            
            try:
                positive_out, negative_out, latent = wan_image_to_video.encode(
                    positive, negative, vae, width, height, frames, 1, loaded_image, None
                )
                logger.info("✅ Image to video encoding successful")
            except Exception as e:
                logger.error(f"❌ Image to video encoding failed: {e}")
                logger.error(f"  Image tensor: {loaded_image.shape} {loaded_image.dtype}")
                logger.error(f"  Parameters: {width}x{height}, {frames} frames")
                raise e
            
            # STAGE 1: High noise model
            logger.info("🎯 STAGE 1: High noise model...")
            model = unet_loader.load_unet("wan2.2_i2v_high_noise_14B_Q6_K.gguf")[0]
            
            # Apply prompt assist LoRA
            if prompt_assist != "none":
                lora_mapping = {
                    "walking to viewers": "walking to viewers_Wan.safetensors",
                    "walking from behind": "walking_from_behind.safetensors",
                    "b3ll13-d8nc3r": "b3ll13-d8nc3r.safetensors"
                }
                
                if prompt_assist in lora_mapping:
                    lora_file = lora_mapping[prompt_assist]
                    logger.info(f"🎭 Applying prompt assist: {lora_file}")
                    model = prompt_assist_lora.load_lora_model_only(model, lora_file, 1.0)[0]
            
            # Apply custom LoRA
            if use_lora and lora_url:
                logger.info("🎨 Processing custom LoRA...")
                lora_path = download_lora_dynamic(lora_url, civitai_token)
                if lora_path:
                    model = custom_lora.load_lora_model_only(model, os.path.basename(lora_path), lora_strength)[0]
                    logger.info(f"✅ Custom LoRA applied: {os.path.basename(lora_path)}")
            
            # Apply LightX2V LoRA
            if use_lightx2v:
                logger.info(f"⚡ Applying LightX2V rank {lightx2v_rank}...")
                lightx2v_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_path:
                    model = lightx2v_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_path), lightx2v_strength
                    )[0]
                    logger.info("✅ LightX2V LoRA applied")
            
            # Apply optimizations
            if use_sage_attention:
                try:
                    model = pathch_sage_attention.patch(model, "auto")[0]
                    logger.info("✅ Sage Attention applied")
                except Exception as e:
                    logger.warning(f"⚠️ Sage Attention failed: {e}")
            
            if rel_l1_thresh > 0:
                try:
                    model = teacache.patch_teacache(
                        model, rel_l1_thresh, 0.2, 1.0, "main_device", "14B"
                    )[0]
                    logger.info(f"✅ TeaCache applied (threshold: {rel_l1_thresh})")
                except Exception as e:
                    logger.warning(f"⚠️ TeaCache failed: {e}")
            
            # High noise sampling
            logger.info(f"🎬 High noise sampling...")
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
            clear_memory_comprehensive()
            
            # STAGE 2: Low noise model
            logger.info("🎯 STAGE 2: Low noise model...")
            model = unet_loader.load_unet("wan2.2_i2v_low_noise_14B_Q6_K.gguf")[0]
            
            # Reapply LoRAs for low noise model
            if prompt_assist != "none" and prompt_assist in lora_mapping:
                model = prompt_assist_lora.load_lora_model_only(model, lora_mapping[prompt_assist], 1.0)[0]
            
            if use_lora and lora_url and lora_path:
                model = custom_lora.load_lora_model_only(model, os.path.basename(lora_path), lora_strength)[0]
            
            # Apply PUSA LoRA
            if use_pusa:
                logger.info(f"🎭 Applying PUSA LoRA...")
                pusa_path = get_lightx2v_lora_path(lightx2v_rank)
                if pusa_path:
                    model = pusa_lora.load_lora_model_only(
                        model, os.path.basename(pusa_path), pusa_strength
                    )[0]
                    logger.info("✅ PUSA LoRA applied")
            
            # Reapply optimizations
            if use_sage_attention:
                try:
                    model = pathch_sage_attention.patch(model, "auto")[0]
                except Exception as e:
                    logger.warning(f"⚠️ Sage Attention (stage 2) failed: {e}")
            
            if rel_l1_thresh > 0:
                try:
                    model = teacache.patch_teacache(
                        model, rel_l1_thresh, 0.2, 1.0, "main_device", "14B"
                    )[0]
                except Exception as e:
                    logger.warning(f"⚠️ TeaCache (stage 2) failed: {e}")
            
            # Low noise sampling
            logger.info(f"🎬 Low noise sampling...")
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
            clear_memory_comprehensive()
            
            # Decode frames
            logger.info("🎨 Decoding latents to frames...")
            decoded = vae_decode.decode(vae, sampled)[0]
            
            del vae
            clear_memory_comprehensive()
            
            # Save video
            logger.info("💾 Saving video với production method...")
            output_path = f"/app/ComfyUI/output/wan22_production_final_{uuid.uuid4().hex[:8]}.mp4"
            
            final_output_path = save_video_production_final(decoded, output_path, fps)
            
            # Frame interpolation (optional)
            if enable_interpolation and interpolation_factor > 1:
                logger.info(f"🔄 PRODUCTION FINAL RIFE interpolation...")
                
                pre_size = os.path.getsize(final_output_path) / (1024 * 1024)
                logger.info(f"📊 Pre-interpolation: {pre_size:.1f}MB")
                
                interpolated_path = apply_rife_interpolation_production_final(
                    final_output_path, interpolation_factor
                )
                
                if interpolated_path != final_output_path and os.path.exists(interpolated_path):
                    post_size = os.path.getsize(interpolated_path) / (1024 * 1024)
                    logger.info(f"📊 Post-interpolation: {post_size:.1f}MB")
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
        clear_memory_comprehensive()

# ================== STORAGE & VALIDATION ==================

def upload_to_minio_enhanced(local_path: str, object_name: str) -> str:
    """📤 Enhanced MinIO upload với comprehensive error handling"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading: {object_name} ({file_size_mb:.1f}MB)")
        
        # Upload với progress monitoring
        start_time = time.time()
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        upload_time = time.time() - start_time
        
        # Generate URL
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        
        logger.info(f"✅ Upload completed: {file_url}")
        logger.info(f"📊 Upload stats: {file_size_mb:.1f}MB in {upload_time:.1f}s")
        
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def validate_input_comprehensive(job_input: dict) -> tuple[bool, str]:
    """🔍 PRODUCTION: Comprehensive input validation"""
    try:
        # Required parameters
        required_params = ["image_url", "positive_prompt"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        # Image URL validation
        image_url = job_input["image_url"]
        try:
            response = requests.head(image_url, timeout=15)
            if response.status_code not in [200, 301, 302, 404]:  # 404 might still work
                logger.warning(f"⚠️ Image URL returned: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Image URL validation warning: {e}")
        
        # Validate dimensions (reasonable limits)
        width = job_input.get("width", 720)
        height = job_input.get("height", 1280)
        if not (256 <= width <= 1536 and 256 <= height <= 1536):
            return False, "Width and height must be between 256 and 1536"
        
        # Validate frames (reasonable limit)
        frames = job_input.get("frames", 65)
        if not (1 <= frames <= 150):
            return False, "Frames must be between 1 and 150"
        
        # Enhanced prompt_assist validation
        prompt_assist = job_input.get("prompt_assist", "none")
        if isinstance(prompt_assist, str):
            prompt_assist_normalized = prompt_assist.lower().strip()
        else:
            prompt_assist_normalized = str(prompt_assist).lower().strip()
        
        # Comprehensive alias mapping
        valid_assists_map = {
            # Standard values
            "none": "none",
            "walking to viewers": "walking to viewers",
            "walking from behind": "walking from behind",
            "b3ll13-d8nc3r": "b3ll13-d8nc3r",
            
            # Common aliases
            "walking_to_viewers": "walking to viewers",
            "walking_from_behind": "walking from behind",
            "walk_to_viewers": "walking to viewers",
            "walk_from_behind": "walking from behind",
            "walkingtoviewers": "walking to viewers",
            "walkingfrombehind": "walking from behind",
            
            # Dance aliases
            "dance": "b3ll13-d8nc3r",
            "dancing": "b3ll13-d8nc3r",
            "dancer": "b3ll13-d8nc3r",
            "belly_dance": "b3ll13-d8nc3r",
            "belly-dance": "b3ll13-d8nc3r",
            "bellydance": "b3ll13-d8nc3r",
            
            # Empty/null handling
            "": "none",
            "null": "none",
            "undefined": "none"
        }
        
        if prompt_assist_normalized not in valid_assists_map:
            valid_display = ["none", "walking to viewers", "walking from behind", "b3ll13-d8nc3r"]
            return False, f"prompt_assist must be one of: {', '.join(valid_display)}"
        
        # Update with normalized value
        job_input["prompt_assist"] = valid_assists_map[prompt_assist_normalized]
        
        # Validate aspect mode
        aspect_mode = job_input.get("aspect_mode", "preserve")
        if aspect_mode not in ["preserve", "crop", "stretch"]:
            return False, "aspect_mode must be one of: preserve, crop, stretch"
        
        # Validate LightX2V rank
        lightx2v_rank = job_input.get("lightx2v_rank", "32")
        if str(lightx2v_rank) not in ["32", "64", "128"]:
            return False, "lightx2v_rank must be one of: 32, 64, 128"
        job_input["lightx2v_rank"] = str(lightx2v_rank)  # Ensure string
        
        # Validate numerical ranges
        cfg_scale = job_input.get("cfg_scale", 1.0)
        if not (0.1 <= cfg_scale <= 15.0):
            return False, "cfg_scale must be between 0.1 and 15.0"
        
        steps = job_input.get("steps", 6)
        if not (1 <= steps <= 50):
            return False, "steps must be between 1 and 50"
        
        high_noise_steps = job_input.get("high_noise_steps", 3)
        if not (1 <= high_noise_steps < steps):
            return False, f"high_noise_steps must be between 1 and {steps-1}"
        
        # Validate sampler
        sampler_name = job_input.get("sampler_name", "euler")
        valid_samplers = [
            "euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", 
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
            "dpmpp_sde", "dpmpp_2m", "ddim"
        ]
        if sampler_name not in valid_samplers:
            return False, f"Invalid sampler. Must be one of: {', '.join(valid_samplers)}"
        
        # Validate interpolation
        interpolation_factor = job_input.get("interpolation_factor", 2)
        if not (2 <= interpolation_factor <= 8):
            return False, "interpolation_factor must be between 2 and 8"
        
        return True, "All parameters valid"
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False, f"Validation error: {str(e)}"

# ================== MAIN HANDLER (PRODUCTION FINAL) ==================

def handler(job):
    """
    🚀 PRODUCTION FINAL: Main RunPod handler
    ✅ Comprehensive CUDA safety, tensor validation, aspect control, RIFE interpolation
    🎯 Supports all popular image sizes và production deployment
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Comprehensive input validation
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
                "error": "ComfyUI system not available",
                "status": "failed",
                "job_id": job_id,
                "system_issue": "comfyui_import_failed"
            }
        
        models_ok, missing_models = verify_models_comprehensive()
        if not models_ok:
            return {
                "error": "Required models missing",
                "missing_models": missing_models,
                "status": "failed",
                "job_id": job_id
            }
        
        # Extract validated parameters
        image_url = job_input["image_url"]
        positive_prompt = job_input["positive_prompt"]
        
        logger.info(f"🚀 Job {job_id}: WAN2.2 PRODUCTION FINAL started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"📝 Prompt: {positive_prompt[:100]}...")
        logger.info(f"⚙️ System: CUDA={CUDA_AVAILABLE}, ComfyUI={COMFYUI_AVAILABLE}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download input image
            image_path = os.path.join(temp_dir, "input_image.jpg")
            
            try:
                logger.info("📥 Downloading input image...")
                response = requests.get(image_url, timeout=60, stream=True)
                response.raise_for_status()
                
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                image_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                logger.info(f"✅ Image downloaded: {image_size_mb:.1f}MB")
                
                # Basic image validation
                try:
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                        logger.info(f"📊 Image dimensions: {img_width}x{img_height}")
                        
                        if img_width < 64 or img_height < 64:
                            return {"error": "Image too small (minimum 64x64)"}
                        if img_width > 4096 or img_height > 4096:
                            return {"error": "Image too large (maximum 4096x4096)"}
                except Exception as e:
                    logger.warning(f"⚠️ Image validation warning: {e}")
                
            except Exception as e:
                return {
                    "error": f"Failed to download image: {str(e)}",
                    "status": "failed",
                    "job_id": job_id
                }
            
            # Generate video với PRODUCTION FINAL method
            logger.info("🎬 Starting PRODUCTION FINAL video generation...")
            generation_start = time.time()
            
            output_path = generate_video_wan22_production_final(
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
            
            # Upload result với enhanced method
            logger.info("📤 Uploading result...")
            output_filename = f"wan22_production_final_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            
            try:
                output_url = upload_to_minio_enhanced(output_path, output_filename)
            except Exception as e:
                return {
                    "error": f"Failed to upload result: {str(e)}",
                    "status": "failed",
                    "job_id": job_id
                }
            
            # Calculate comprehensive statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            duration_seconds = job_input.get("frames", 65) / job_input.get("fps", 16)
            
            # Check if interpolated
            is_interpolated = (job_input.get("enable_interpolation", False) and 
                             "_rife_" in os.path.basename(output_path))
            actual_interpolation_factor = (job_input.get("interpolation_factor", 2) 
                                         if is_interpolated else 1)
            
            actual_seed = job_input.get("seed", 0) if job_input.get("seed", 0) != 0 else "auto-generated"
            
            logger.info(f"✅ Job {job_id} COMPLETED SUCCESSFULLY!")
            logger.info(f"⏱️ Total: {total_time:.1f}s (generation: {generation_time:.1f}s)")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB, duration: {duration_seconds:.1f}s")
            logger.info(f"🎬 Features used: aspect={job_input.get('aspect_mode', 'preserve')}, interpolation={is_interpolated}")
            
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
                    "interpolated": is_interpolated,
                    "interpolation_factor": actual_interpolation_factor,
                    "aspect_mode": job_input.get("aspect_mode", "preserve"),
                    "auto_720p": job_input.get("auto_720p", False)
                },
                
                "generation_params": {
                    "positive_prompt": job_input["positive_prompt"],
                    "negative_prompt": (job_input.get("negative_prompt", "")[:80] + "..." 
                                      if len(job_input.get("negative_prompt", "")) > 80 
                                      else job_input.get("negative_prompt", "")),
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
                        "url": job_input.get("lora_url", None),
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
                        "workflow_version": "PRODUCTION_FINAL_v4.0",
                        "tensor_fixes_applied": True,
                        "aspect_control_enhanced": True,
                        "rife_interpolation_fixed": True
                    }
                },
                
                "status": "completed"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": f"Generation failed: {error_msg}",
            "status": "failed",
            "job_id": job_id,
            "processing_time_seconds": round(time.time() - start_time, 2),
            "system_info": {
                "cuda_available": CUDA_AVAILABLE,
                "comfyui_available": COMFYUI_AVAILABLE,
                "error_category": "generation_error"
            }
        }
        
    finally:
        clear_memory_comprehensive()

# ================== HEALTH CHECK (PRODUCTION FINAL) ==================

def health_check():
    """🏥 PRODUCTION FINAL: Comprehensive system health check"""
    try:
        issues = []
        warnings = []
        
        # CUDA check
        if not torch.cuda.is_available():
            issues.append("CUDA not available")
        elif not CUDA_AVAILABLE:
            warnings.append("CUDA compatibility issues detected")
        
        # GPU memory check
        if torch.cuda.is_available():
            try:
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if memory_gb < 8:
                    warnings.append(f"Low GPU memory: {memory_gb:.1f}GB")
                elif memory_gb < 16:
                    warnings.append(f"Moderate GPU memory: {memory_gb:.1f}GB")
                
                # Test basic GPU operations
                test_tensor = torch.ones(100, device='cuda')
                test_result = test_tensor.sum()
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
            except Exception as e:
                issues.append(f"GPU test failed: {str(e)}")
        
        # ComfyUI check
        if not COMFYUI_AVAILABLE:
            issues.append("ComfyUI modules not available")
        
        # Models check
        models_ok, missing = verify_models_comprehensive()
        if not models_ok:
            issues.append(f"Missing {len(missing)} required models")
        
        # Storage check
        if not minio_client:
            issues.append("MinIO storage not available")
        else:
            try:
                # Test MinIO connectivity
                buckets = minio_client.list_buckets()
                logger.debug(f"MinIO buckets accessible: {len(buckets)}")
            except Exception as e:
                warnings.append(f"MinIO connectivity issue: {str(e)}")
        
        # Disk space check
        try:
            import shutil
            total, used, free = shutil.disk_usage("/app")
            free_gb = free / (1024**3)
            if free_gb < 5:
                issues.append(f"Low disk space: {free_gb:.1f}GB free")
            elif free_gb < 10:
                warnings.append(f"Moderate disk space: {free_gb:.1f}GB free")
        except Exception as e:
            warnings.append(f"Disk space check failed: {str(e)}")
        
        # RIFE check
        rife_script = "/app/Practical-RIFE/inference_video.py"
        if not os.path.exists(rife_script):
            warnings.append("RIFE interpolation not available")
        
        # Summary
        if issues:
            return False, f"Critical issues: {'; '.join(issues)}" + (f" | Warnings: {'; '.join(warnings)}" if warnings else "")
        elif warnings:
            return True, f"Operational with warnings: {'; '.join(warnings)}"
        else:
            return True, "All systems operational - PRODUCTION READY"
            
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

# ================== STARTUP DIAGNOSTICS ==================

def startup_diagnostics():
    """🔍 Comprehensive startup diagnostics"""
    logger.info("🔍 Running startup diagnostics...")
    
    # System info
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🐍 Python: {sys.version.split()[0]}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            logger.info(f"🎯 GPU {i}: {props.name}, {memory_gb:.1f}GB, Compute {props.major}.{props.minor}")
    else:
        logger.warning("⚠️ No CUDA devices available")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"💾 RAM: {memory.total / 1e9:.1f}GB total, {memory.available / 1e9:.1f}GB available")
    except ImportError:
        logger.info("💾 RAM info not available (psutil not installed)")
    
    # Disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/app")
        logger.info(f"💿 Disk: {total / 1e9:.1f}GB total, {free / 1e9:.1f}GB free")
    except Exception as e:
        logger.warning(f"⚠️ Disk info failed: {e}")
    
    # Feature summary
    features = []
    if CUDA_AVAILABLE:
        features.append("CUDA")
    if COMFYUI_AVAILABLE:
        features.append("ComfyUI")
    if minio_client:
        features.append("MinIO")
    if os.path.exists("/app/Practical-RIFE/inference_video.py"):
        features.append("RIFE")
    
    logger.info(f"✨ Available features: {', '.join(features) if features else 'None'}")

# ================== MAIN ENTRY POINT (PRODUCTION FINAL) ==================

if __name__ == "__main__":
    logger.info("🚀 Starting WAN2.2 LightX2V PRODUCTION FINAL Serverless Worker...")
    
    try:
        # Comprehensive startup diagnostics
        startup_diagnostics()
        
        # Run health check
        health_ok, health_msg = health_check()
        
        logger.info(f"📊 System Health: {health_msg}")
        
        if not health_ok:
            logger.error("❌ Critical health check failures detected!")
            logger.error("💥 System cannot start in production mode!")
            
            # Try to continue with limited functionality
            logger.info("🔄 Attempting to start with limited functionality...")
        
        # Feature availability summary
        logger.info("📋 PRODUCTION FINAL Features:")
        logger.info("  ✅ Enhanced CUDA Compatibility Fixes")
        logger.info("  ✅ Comprehensive Tensor Shape Validation")
        logger.info("  ✅ Advanced Aspect Ratio Control (preserve/crop/stretch)")
        logger.info("  ✅ Multi-Size Image Support (64x64 to 4096x4096)")
        logger.info("  ✅ RIFE Frame Interpolation (FIXED)")
        logger.info("  ✅ LightX2V LoRA Support (rank 32/64/128)")
        logger.info("  ✅ Custom LoRA Download (HuggingFace/CivitAI)")
        logger.info("  ✅ Advanced Optimizations (Sage Attention, TeaCache)")
        logger.info("  ✅ Production-Grade Error Handling")
        logger.info("  ✅ Comprehensive Input Validation")
        logger.info("  ✅ Enhanced Storage Management")
        logger.info("  ✅ Detailed Performance Monitoring")
        
        # System readiness status
        readiness_indicators = {
            "CUDA": "✅" if CUDA_AVAILABLE else "❌",
            "ComfyUI": "✅" if COMFYUI_AVAILABLE else "❌",
            "Storage": "✅" if minio_client else "❌",
            "Models": "✅" if verify_models_comprehensive()[0] else "❌",
            "RIFE": "✅" if os.path.exists("/app/Practical-RIFE/inference_video.py") else "⚠️"
        }
        
        readiness_summary = " | ".join([f"{k}: {v}" for k, v in readiness_indicators.items()])
        logger.info(f"🎯 System Readiness: {readiness_summary}")
        
        if all(v == "✅" for k, v in readiness_indicators.items() if k != "RIFE"):
            logger.info("🎬 PRODUCTION FINAL READY - All critical systems operational!")
        elif COMFYUI_AVAILABLE and verify_models_comprehensive()[0]:
            logger.info("🎬 BASIC READY - Core functionality available!")
        else:
            logger.warning("⚠️ LIMITED FUNCTIONALITY - Some systems unavailable!")
        
        # Performance tips
        if CUDA_AVAILABLE:
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_gb >= 24:
                logger.info("🚀 HIGH PERFORMANCE MODE - Optimal GPU memory available")
            elif memory_gb >= 16:
                logger.info("⚡ PERFORMANCE MODE - Good GPU memory available")
            elif memory_gb >= 8:
                logger.info("🔧 EFFICIENCY MODE - Limited GPU memory, using optimizations")
            else:
                logger.warning("⚠️ LOW MEMORY MODE - Consider enabling aggressive optimizations")
        
        # Final startup message
        logger.info("=" * 80)
        logger.info("🎉 WAN2.2 PRODUCTION FINAL Worker Started Successfully!")
        logger.info("📞 Ready to process video generation requests...")
        logger.info("🔗 RunPod integration active")
        logger.info("=" * 80)
        
        # Start RunPod serverless worker
        runpod.serverless.start({"handler": handler})
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Critical startup failure: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.error("💥 SYSTEM CANNOT START")
        logger.error("🔧 Please check:")
        logger.error("  - CUDA drivers and compatibility")
        logger.error("  - ComfyUI installation and dependencies")
        logger.error("  - Model files availability")
        logger.error("  - Storage connectivity")
        logger.error("  - System resources (memory, disk space)")
        
        sys.exit(1)
