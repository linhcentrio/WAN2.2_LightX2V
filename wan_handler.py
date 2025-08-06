#!/usr/bin/env python3

"""
RunPod Serverless Handler cho WAN2.2 LightX2V Q6_K - FINAL OPTIMIZED VERSION
✨ Features: Aspect Ratio Control, RIFE Interpolation Fixed, Enhanced Processing
📋 Based on wan22_Lightx2v.ipynb workflow - Production Ready
🔧 All fixes, optimizations và stability improvements included
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

# ================== LOGGING CONFIGURATION ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== SYSTEM PATHS ==================
sys.path.insert(0, '/app/ComfyUI')
sys.path.insert(0, '/app/Practical-RIFE')

# ================== COMFYUI IMPORTS ==================
try:
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
    
    # Import folder paths for proper ComfyUI integration
    import folder_paths
    
    logger.info("✅ All ComfyUI modules imported successfully")
    COMFYUI_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"❌ ComfyUI import error: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    COMFYUI_AVAILABLE = False

# ================== MINIO CONFIGURATION ==================
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client với error handling
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

# ================== MODEL CONFIGURATIONS ==================
MODEL_CONFIGS = {
    # Q6_K DIT Models (matching notebook defaults)
    "dit_model_high": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
    "dit_model_low": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf",
    
    # Supporting models (exact filenames from notebook)
    "text_encoder": "/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "/app/ComfyUI/models/vae/wan_2.1_vae.safetensors",
    "clip_vision": "/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
    
    # LightX2V LoRAs theo rank (matching notebook)
    "lightx2v_rank_32": "/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
    "lightx2v_rank_64": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
    "lightx2v_rank_128": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
    
    # Built-in LoRAs với tên file chính xác từ notebook
    "walking_to_viewers": "/app/ComfyUI/models/loras/walking to viewers_Wan.safetensors",
    "walking_from_behind": "/app/ComfyUI/models/loras/walking_from_behind.safetensors",
    "dancing": "/app/ComfyUI/models/loras/b3ll13-d8nc3r.safetensors",
    "pusa_lora": "/app/ComfyUI/models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors",
    "rotate_lora": "/app/ComfyUI/models/loras/rotate_20_epochs.safetensors"
}

# Global model cache để tối ưu memory
model_cache = {}

# ================== PYTORCH OPTIMIZATIONS ==================
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # Enable Flash Attention trong PyTorch 2.6.0
    torch.backends.cuda.flash_sdp_enabled = True
    torch.backends.cuda.mem_efficient_sdp_enabled = True
    logger.info("✅ PyTorch 2.6.0 optimizations enabled")
except Exception as e:
    logger.warning(f"⚠️ PyTorch optimizations partially failed: {e}")

# ================== UTILITY FUNCTIONS ==================

def verify_models() -> tuple[bool, list]:
    """
    🔍 Verify tất cả models cần thiết có tồn tại
    Returns: (success, missing_models)
    """
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
                missing_models.append(f"{name}: {path} (error reading)")
        else:
            missing_models.append(f"{name}: {path}")
            logger.error(f"❌ Missing: {name} at {path}")
    
    if missing_models:
        logger.error(f"❌ Missing {len(missing_models)}/{len(MODEL_CONFIGS)} models")
        for model in missing_models:
            logger.error(f"  - {model}")
        return False, missing_models
    else:
        logger.info(f"✅ All {len(existing_models)} models verified! Total: {total_size:.1f}MB")
        return True, []

def download_lora_dynamic(lora_url: str, civitai_token: str = None) -> str:
    """
    🎨 Download LoRA từ HuggingFace hoặc CivitAI
    Supports: HuggingFace, CivitAI, Direct URLs
    """
    try:
        lora_dir = "/app/ComfyUI/models/loras"
        os.makedirs(lora_dir, exist_ok=True)
        
        logger.info(f"🎨 Downloading LoRA from: {lora_url}")
        
        if "huggingface.co" in lora_url:
            # HuggingFace download
            parts = lora_url.split("/")
            if len(parts) >= 7 and "resolve" in parts:
                username = parts[3]
                repo = parts[4]
                filename = parts[-1]
                local_path = os.path.join(lora_dir, filename)
                
                logger.info(f"📥 Downloading from HuggingFace: {username}/{repo}/{filename}")
                downloaded_path = hf_hub_download(
                    repo_id=f"{username}/{repo}",
                    filename=filename,
                    local_dir=lora_dir,
                    force_download=True
                )
                logger.info(f"✅ HuggingFace LoRA downloaded: {filename}")
                return downloaded_path
                
        elif "civitai.com" in lora_url:
            # CivitAI download
            try:
                if "/models/" in lora_url:
                    model_id = lora_url.split("/models/")[1].split("?")[0].split("/")[0]
                else:
                    raise ValueError("Invalid CivitAI URL format")
                
                headers = {}
                if civitai_token:
                    headers["Authorization"] = f"Bearer {civitai_token}"
                
                api_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
                logger.info(f"📥 Downloading from CivitAI: model_id={model_id}")
                
                response = requests.get(api_url, headers=headers, timeout=300, stream=True)
                response.raise_for_status()
                
                # Generate unique filename
                timestamp = int(time.time())
                filename = f"civitai_model_{model_id}_{timestamp}.safetensors"
                local_path = os.path.join(lora_dir, filename)
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                                    logger.info(f"📥 Progress: {progress:.1f}% ({downloaded/1024/1024:.1f}MB)")
                
                logger.info(f"✅ CivitAI LoRA downloaded: {filename} ({downloaded/1024/1024:.1f}MB)")
                return local_path
                
            except Exception as e:
                logger.error(f"❌ CivitAI download failed: {e}")
                return None
        else:
            # Direct download
            filename = os.path.basename(urlparse(lora_url).path)
            if not filename or not filename.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                filename = f"downloaded_lora_{int(time.time())}.safetensors"
            
            local_path = os.path.join(lora_dir, filename)
            logger.info(f"📥 Direct download: {filename}")
            
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
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def clear_memory():
    """🧹 Enhanced memory cleanup matching notebook"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Additional CUDA memory cleanup
        try:
            torch.cuda.synchronize()
        except:
            pass

def get_lightx2v_lora_path(lightx2v_rank: str) -> str:
    """📍 Get LightX2V LoRA path theo rank (exact from notebook)"""
    rank_mapping = {
        "32": "lightx2v_rank_32",
        "64": "lightx2v_rank_64",
        "128": "lightx2v_rank_128"
    }
    config_key = rank_mapping.get(lightx2v_rank, "lightx2v_rank_32")
    return MODEL_CONFIGS.get(config_key)

# ================== ASPECT RATIO PROCESSING ==================

def calculate_aspect_preserving_dimensions(original_width, original_height, target_width, target_height):
    """
    📐 Tính toán kích thước giữ nguyên aspect ratio với padding
    """
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height
    
    if original_ratio > target_ratio:
        # Ảnh rộng hơn -> padding top/bottom
        new_width = target_width
        new_height = int(target_width / original_ratio)
    else:
        # Ảnh cao hơn -> padding left/right  
        new_height = target_height
        new_width = int(target_height * original_ratio)
    
    # Đảm bảo chia hết cho 8 cho video encoding
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height

def calculate_crop_dimensions(original_width, original_height, target_width, target_height):
    """
    ✂️ Tính toán crop thông minh để giữ nguyên tỉ lệ
    """
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    
    if original_ratio > target_ratio:
        # Ảnh rộng hơn -> crop width
        crop_height = original_height
        crop_width = int(original_height * target_ratio)
        crop_x = (original_width - crop_width) // 2
        crop_y = 0
    else:
        # Ảnh cao hơn -> crop height  
        crop_width = original_width
        crop_height = int(original_width / target_ratio)
        crop_x = 0
        crop_y = (original_height - crop_height) // 2
    
    return crop_x, crop_y, crop_width, crop_height

def process_image_with_aspect_control(
    image_path: str,
    target_width: int = 720,
    target_height: int = 1280,
    aspect_mode: str = "preserve",  # "preserve", "crop", "stretch"
    auto_720p: bool = False
):
    """
    🖼️ Enhanced image processing với aspect ratio control
    
    Args:
        aspect_mode: 
            - "preserve": Giữ nguyên tỉ lệ, padding nếu cần
            - "crop": Crop thông minh để fit target ratio
            - "stretch": Stretch trực tiếp (backward compatible)
        auto_720p: Tự động điều chỉnh về chuẩn 720p
    """
    try:
        # Initialize nodes
        load_image = LoadImage()
        image_scaler = ImageScale()
        
        # Load ảnh gốc
        loaded_image = load_image.load_image(image_path)[0]
        
        # Lấy kích thước gốc
        if loaded_image.ndim == 4:
            _, orig_height, orig_width, _ = loaded_image.shape
        else:
            orig_height, orig_width, _ = loaded_image.shape
        
        logger.info(f"🖼️ Original image: {orig_width}x{orig_height}")
        
        # Auto 720p detection
        if auto_720p:
            # Tự động chọn orientation dựa trên ảnh gốc
            if orig_width > orig_height:
                # Landscape -> 720p horizontal
                target_width, target_height = 1280, 720
            else:
                # Portrait -> 720p vertical  
                target_width, target_height = 720, 1280
            logger.info(f"📱 Auto 720p mode: {target_width}x{target_height}")
        
        if aspect_mode == "preserve":
            # Giữ nguyên aspect ratio với padding
            new_width, new_height = calculate_aspect_preserving_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            # Scale về kích thước tính toán
            scaled_image = image_scaler.upscale(
                loaded_image, "lanczos", new_width, new_height, "disabled"
            )[0]
            
            # Padding để đạt target size nếu cần
            if new_width != target_width or new_height != target_height:
                # Tạo tensor với padding
                pad_x = (target_width - new_width) // 2
                pad_y = (target_height - new_height) // 2
                
                # Convert to tensor format for padding
                if scaled_image.ndim == 3:
                    scaled_image = scaled_image.unsqueeze(0)  # Add batch dim
                
                # Padding: (left, right, top, bottom)
                padded = F.pad(
                    scaled_image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                    (pad_x, target_width - new_width - pad_x, 
                     pad_y, target_height - new_height - pad_y),
                    mode='constant', value=0
                )
                
                final_image = padded.permute(0, 2, 3, 1)[0]  # BCHW -> HWC
            else:
                final_image = scaled_image
                
            logger.info(f"📐 Preserve mode: {orig_width}x{orig_height} -> {new_width}x{new_height} -> {target_width}x{target_height}")
            
        elif aspect_mode == "crop":
            # Smart crop để giữ nguyên aspect ratio
            crop_x, crop_y, crop_width, crop_height = calculate_crop_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            # Crop ảnh gốc trước
            if loaded_image.ndim == 3:
                cropped = loaded_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
            else:
                cropped = loaded_image[0, crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
                cropped = cropped.unsqueeze(0)
            
            # Scale về target size
            final_image = image_scaler.upscale(
                cropped, "lanczos", target_width, target_height, "disabled"
            )[0]
            
            logger.info(f"✂️ Crop mode: {orig_width}x{orig_height} -> crop({crop_width}x{crop_height}) -> {target_width}x{target_height}")
            
        else:  # stretch
            # Stretch trực tiếp (workflow cũ)
            final_image = image_scaler.upscale(
                loaded_image, "lanczos", target_width, target_height, "disabled"
            )[0]
            
            logger.info(f"🔄 Stretch mode: {orig_width}x{orig_height} -> {target_width}x{target_height}")
        
        return final_image, target_width, target_height
        
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

# ================== RIFE INTERPOLATION (FIXED) ==================

def apply_rife_interpolation_production(video_path: str, interpolation_factor: int = 2) -> str:
    """
    🔧 PRODUCTION GRADE: RIFE frame interpolation với comprehensive error handling
    ✅ Enhanced output detection, path management, debugging capabilities
    """
    try:
        logger.info(f"🔄 Applying RIFE interpolation (factor: {interpolation_factor}) - PRODUCTION VERSION...")
        
        # Validate input file
        if not os.path.exists(video_path):
            logger.error(f"❌ Input video not found: {video_path}")
            return video_path
        
        original_size = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"📊 Original video: {original_size:.1f}MB")
        
        # Check if RIFE script exists
        rife_script = "/app/Practical-RIFE/inference_video.py"
        if not os.path.exists(rife_script):
            logger.error(f"❌ RIFE script not found: {rife_script}")
            return video_path
        
        # Create dedicated output directories
        rife_output_dir = "/app/rife_output"
        rife_temp_dir = "/app/rife_temp"
        os.makedirs(rife_output_dir, exist_ok=True)
        os.makedirs(rife_temp_dir, exist_ok=True)
        
        # Generate specific output filename
        input_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{input_basename}_rife_{interpolation_factor}x.mp4"
        final_output_path = os.path.join(rife_output_dir, output_filename)
        
        # Copy input video to RIFE directory để ensure accessibility
        rife_input_path = os.path.join("/app/Practical-RIFE", os.path.basename(video_path))
        shutil.copy2(video_path, rife_input_path)
        
        logger.info(f"📁 RIFE setup:")
        logger.info(f"  Input: {rife_input_path}")
        logger.info(f"  Output target: {final_output_path}")
        
        # Setup enhanced environment
        env = os.environ.copy()
        env.update({
            "XDG_RUNTIME_DIR": "/tmp",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "FFMPEG_LOGLEVEL": "quiet",
            "CUDA_VISIBLE_DEVICES": "0"
        })
        
        # Build optimized RIFE command
        cmd = [
            "python3",
            "inference_video.py",
            f"--multi={interpolation_factor}",
            f"--video={os.path.basename(video_path)}",
            "--scale=1.0",
            "--fps=30",
            "--png=False",
            "--UHD=False",
            "--crop=0,0,0,0"
        ]
        
        logger.info(f"🔧 RIFE command: {' '.join(cmd)}")
        
        # Execute RIFE với comprehensive monitoring
        original_cwd = os.getcwd()
        os.chdir("/app/Practical-RIFE")
        
        try:
            # Clear existing outputs
            cleanup_patterns = ["*_interpolated.mp4", "*_rife.mp4", "*_output.mp4"]
            for pattern in cleanup_patterns:
                for file in Path(".").glob(pattern):
                    try:
                        file.unlink()
                        logger.info(f"🗑️ Cleared: {file}")
                    except:
                        pass
            
            # Run RIFE với timeout và monitoring
            logger.info("🚀 Starting RIFE interpolation process...")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minutes timeout
                check=False
            )
            
            execution_time = time.time() - start_time
            logger.info(f"⏱️ RIFE execution time: {execution_time:.1f}s")
            
            # Enhanced result logging
            logger.info(f"📊 RIFE process completed:")
            logger.info(f"  Return code: {result.returncode}")
            if result.stdout:
                stdout_preview = result.stdout[:1000] + ("..." if len(result.stdout) > 1000 else "")
                logger.info(f"  Stdout: {stdout_preview}")
            if result.stderr:
                stderr_preview = result.stderr[:1000] + ("..." if len(result.stderr) > 1000 else "")
                logger.info(f"  Stderr: {stderr_preview}")
            
        except subprocess.TimeoutExpired:
            logger.error("❌ RIFE process timed out (15 minutes)")
            return video_path
        except Exception as e:
            logger.error(f"❌ RIFE subprocess failed: {e}")
            return video_path
        finally:
            # Restore directory và cleanup
            os.chdir(original_cwd)
            try:
                if os.path.exists(rife_input_path):
                    os.remove(rife_input_path)
            except:
                pass
        
        # 🔍 COMPREHENSIVE OUTPUT DETECTION
        logger.info("🔍 Searching for RIFE output files...")
        
        potential_outputs = []
        rife_dir = "/app/Practical-RIFE"
        input_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Enhanced search patterns
        search_patterns = [
            # Standard patterns
            f"{input_name}_{interpolation_factor}X.mp4",
            f"{input_name}_{interpolation_factor}x.mp4",
            f"{input_name}_interpolated.mp4",
            f"{input_name}_rife.mp4",
            f"{input_name}_output.mp4",
            
            # Alternative patterns
            f"output_{interpolation_factor}x.mp4",
            f"result_{interpolation_factor}x.mp4",
            "output.mp4",
            "result.mp4",
            
            # Basename variations
            f"{os.path.basename(video_path).replace('.mp4', f'_{interpolation_factor}X.mp4')}",
            f"{os.path.basename(video_path).replace('.mp4', '_interpolated.mp4')}",
            f"{os.path.basename(video_path).replace('.mp4', '_output.mp4')}"
        ]
        
        # Search by patterns
        for pattern in search_patterns:
            potential_path = os.path.join(rife_dir, pattern)
            if os.path.exists(potential_path):
                potential_outputs.append(potential_path)
                logger.info(f"🔍 Pattern match: {pattern}")
        
        # Search by recent modification time
        current_time = time.time()
        try:
            for file in os.listdir(rife_dir):
                if file.endswith('.mp4') and file != os.path.basename(video_path):
                    file_path = os.path.join(rife_dir, file)
                    # Check if created/modified in last 20 minutes
                    if current_time - max(os.path.getctime(file_path), os.path.getmtime(file_path)) < 1200:
                        if file_path not in potential_outputs:
                            potential_outputs.append(file_path)
                            logger.info(f"🔍 Recent file: {file}")
        except Exception as e:
            logger.warning(f"⚠️ Error scanning directory: {e}")
        
        # Analyze và select best output
        selected_output = None
        if potential_outputs:
            logger.info(f"📊 Analyzing {len(potential_outputs)} potential outputs:")
            
            # Create candidate list với metadata
            candidates = []
            for path in potential_outputs:
                try:
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    mod_time = os.path.getmtime(path)
                    # Score based on size (should be larger for interpolation) và recency
                    size_score = min(size_mb / original_size, 3.0)  # Cap at 3x original
                    recency_score = max(0, (1200 - (current_time - mod_time)) / 1200)  # Within 20 mins
                    total_score = size_score * 0.7 + recency_score * 0.3
                    
                    candidates.append({
                        'path': path,
                        'size_mb': size_mb,
                        'score': total_score,
                        'mod_time': mod_time
                    })
                    
                    logger.info(f"  📄 {os.path.basename(path)}: {size_mb:.1f}MB, score: {total_score:.2f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error analyzing {path}: {e}")
            
            # Select highest scoring candidate that meets minimum requirements
            if candidates:
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                for candidate in candidates:
                    # Validate candidate
                    if (candidate['size_mb'] >= original_size * 0.5 and  # At least 50% of original
                        candidate['size_mb'] <= original_size * 10):    # Not unreasonably large
                        selected_output = candidate['path']
                        logger.info(f"✅ Selected best candidate: {os.path.basename(selected_output)} ({candidate['size_mb']:.1f}MB)")
                        break
        
        # Process selected output
        if selected_output and os.path.exists(selected_output):
            try:
                # Move to final location
                shutil.move(selected_output, final_output_path)
                
                final_size = os.path.getsize(final_output_path) / (1024 * 1024)
                size_ratio = final_size / original_size
                
                logger.info(f"✅ RIFE interpolation successful!")
                logger.info(f"📊 Results:")
                logger.info(f"  Original: {original_size:.1f}MB")
                logger.info(f"  Interpolated: {final_size:.1f}MB ({size_ratio:.1f}x)")
                logger.info(f"  Processing time: {execution_time:.1f}s")
                logger.info(f"  Output: {final_output_path}")
                
                return final_output_path
                
            except Exception as e:
                logger.error(f"❌ Failed to move output file: {e}")
                return video_path
        else:
            # Comprehensive debugging khi không tìm thấy output
            logger.warning("🔍 COMPREHENSIVE DEBUGGING - No valid output found")
            logger.warning("📁 Current RIFE directory contents:")
            
            try:
                files_found = []
                for item in os.listdir(rife_dir):
                    item_path = os.path.join(rife_dir, item)
                    if os.path.isfile(item_path):
                        size_mb = os.path.getsize(item_path) / (1024 * 1024)
                        mod_time = time.ctime(os.path.getmtime(item_path))
                        create_time = time.ctime(os.path.getctime(item_path))
                        
                        files_found.append(f"  📄 {item}: {size_mb:.1f}MB")
                        files_found.append(f"      Modified: {mod_time}")
                        files_found.append(f"      Created: {create_time}")
                
                for file_info in files_found:
                    logger.warning(file_info)
                    
            except Exception as e:
                logger.error(f"❌ Error listing directory for debugging: {e}")
            
            logger.warning(f"⚠️ RIFE interpolation failed - using original video")
            return video_path
            
    except Exception as e:
        logger.error(f"❌ RIFE interpolation critical error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return video_path

# ================== VIDEO SAVING ==================

def save_video_optimized(frames_tensor, output_path, fps=16):
    """
    🎬 PRODUCTION GRADE video saving function
    ✅ Multiple encoding methods, comprehensive validation, error recovery
    """
    try:
        logger.info(f"🎬 Saving video with PRODUCTION OPTIMIZED method...")
        logger.info(f"📍 Output path: {output_path}")
        
        # Validate input
        if frames_tensor is None:
            raise ValueError("Frames tensor is None")
        
        # Convert tensor to numpy - exactly like notebook
        if torch.is_tensor(frames_tensor):
            frames_np = frames_tensor.detach().cpu().float().numpy()
        else:
            frames_np = np.array(frames_tensor, dtype=np.float32)
        
        logger.info(f"📊 Input tensor shape: {frames_np.shape}, dtype: {frames_np.dtype}")
        
        # Handle batch dimension if present
        if frames_np.ndim == 5 and frames_np.shape[0] == 1:
            frames_np = frames_np[0]  # Remove batch dimension
            logger.info(f"📊 After batch removal: {frames_np.shape}")
        
        # Convert to uint8 (0-255 range)
        if frames_np.dtype != np.uint8:
            logger.info("🔄 Converting to uint8 format...")
            # Ensure values are in [0, 1] range first
            if frames_np.max() <= 1.0:
                frames_np = (frames_np * 255.0).astype(np.uint8)
            else:
                frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)
        
        logger.info(f"📊 Final tensor stats:")
        logger.info(f"  Shape: {frames_np.shape}")
        logger.info(f"  Dtype: {frames_np.dtype}")
        logger.info(f"  Value range: [{frames_np.min()}, {frames_np.max()}]")
        
        # Validate dimensions
        if len(frames_np.shape) != 4:
            raise ValueError(f"Invalid shape: {frames_np.shape} (expected 4D: [frames, height, width, channels])")
        
        num_frames, height, width, channels = frames_np.shape
        
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
        
        logger.info(f"📊 Final video specs: {num_frames} frames, {width}x{height}, {channels} channels")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Multiple encoding strategies
        encoding_strategies = [
            # Strategy 1: Standard imageio
            {
                'name': 'Standard',
                'params': {'fps': fps}
            },
            # Strategy 2: High quality
            {
                'name': 'High Quality',
                'params': {
                    'fps': fps,
                    'codec': 'libx264',
                    'pixelformat': 'yuv420p',
                    'quality': 8
                }
            },
            # Strategy 3: Compatibility mode
            {
                'name': 'Compatibility',
                'params': {
                    'fps': fps,
                    'codec': 'libx264',
                    'pixelformat': 'yuv420p',
                    'bitrate': '2M'
                }
            }
        ]
        
        # Try encoding strategies
        for strategy in encoding_strategies:
            try:
                logger.info(f"🎥 Attempting {strategy['name']} encoding...")
                
                with imageio.get_writer(output_path, **strategy['params']) as writer:
                    for i, frame in enumerate(frames_np):
                        writer.append_data(frame)
                        
                        # Progress logging
                        if (i + 1) % max(1, num_frames // 10) == 0:
                            progress = ((i + 1) / num_frames) * 100
                            logger.info(f"📹 Progress: {progress:.1f}% ({i + 1}/{num_frames})")
                
                # Verify output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    logger.info(f"✅ Video saved successfully with {strategy['name']} encoding!")
                    logger.info(f"📊 Final stats:")
                    logger.info(f"  File: {output_path}")
                    logger.info(f"  Size: {file_size_mb:.2f}MB")
                    logger.info(f"  Specs: {num_frames} frames @ {fps}fps, {width}x{height}")
                    
                    return output_path
                else:
                    logger.warning(f"⚠️ {strategy['name']} encoding produced empty file")
                    
            except Exception as e:
                logger.warning(f"⚠️ {strategy['name']} encoding failed: {e}")
                continue
        
        # If all strategies failed
        raise RuntimeError("All encoding strategies failed")
        
    except Exception as e:
        logger.error(f"❌ Video saving failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

# ================== MAIN VIDEO GENERATION ==================

def generate_video_wan22_complete(image_path: str, **kwargs) -> str:
    """
    🎬 PRODUCTION GRADE: Complete WAN2.2 video generation
    ✅ Enhanced error handling, aspect control, optimizations, RIFE support
    """
    try:
        logger.info("🎬 Starting WAN2.2 PRODUCTION generation...")
        
        # ================ PARAMETER EXTRACTION ================
        # Extract parameters với comprehensive defaults
        positive_prompt = kwargs.get('positive_prompt', '')
        negative_prompt = kwargs.get('negative_prompt', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
        
        # Enhanced aspect ratio control
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
        
        # LoRA configurations
        use_lora = kwargs.get('use_lora', False)
        lora_url = kwargs.get('lora_url', None)
        lora_strength = kwargs.get('lora_strength', 1.0)
        civitai_token = kwargs.get('civitai_token', None)
        use_lora2 = kwargs.get('use_lora2', False)
        lora2_url = kwargs.get('lora2_url', None)
        lora2_strength = kwargs.get('lora2_strength', 1.0)
        use_lora3 = kwargs.get('use_lora3', False)
        lora3_url = kwargs.get('lora3_url', None)
        lora3_strength = kwargs.get('lora3_strength', 1.0)
        
        # LightX2V configuration
        use_lightx2v = kwargs.get('use_lightx2v', True)
        lightx2v_rank = kwargs.get('lightx2v_rank', '32')
        lightx2v_strength = kwargs.get('lightx2v_strength', 3.0)
        
        # PUSA configuration
        use_pusa = kwargs.get('use_pusa', True)
        pusa_strength = kwargs.get('pusa_strength', 1.5)
        
        # Optimization settings
        use_sage_attention = kwargs.get('use_sage_attention', True)
        rel_l1_thresh = kwargs.get('rel_l1_thresh', 0.0)
        start_percent = kwargs.get('start_percent', 0.2)
        end_percent = kwargs.get('end_percent', 1.0)
        
        # Flow shift settings
        enable_flow_shift = kwargs.get('enable_flow_shift', True)
        flow_shift = kwargs.get('flow_shift', 8.0)
        enable_flow_shift2 = kwargs.get('enable_flow_shift2', True)
        flow_shift2 = kwargs.get('flow_shift2', 8.0)
        
        # Advanced optimizations
        use_nag = kwargs.get('use_nag', False)
        nag_strength = kwargs.get('nag_strength', 11.0)
        nag_scale1 = kwargs.get('nag_scale1', 0.25)
        nag_scale2 = kwargs.get('nag_scale2', 2.5)
        use_clip_vision = kwargs.get('use_clip_vision', False)
        
        # Frame interpolation
        enable_interpolation = kwargs.get('enable_interpolation', False)
        interpolation_factor = kwargs.get('interpolation_factor', 2)
        
        # Generate seed if auto
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
        
        # Log configuration
        logger.info(f"🎯 Generation Configuration:")
        logger.info(f"  📐 Resolution: {width}x{height} (mode: {aspect_mode}, auto720p: {auto_720p})")
        logger.info(f"  🎬 Animation: {frames} frames @ {fps}fps")
        logger.info(f"  ⚙️ Sampling: {steps} steps (high noise: {high_noise_steps}), CFG: {cfg_scale}")
        logger.info(f"  🌱 Seed: {seed}")
        logger.info(f"  🎭 Prompt assist: {prompt_assist}")
        logger.info(f"  ⚡ LightX2V: {use_lightx2v} (rank: {lightx2v_rank}, strength: {lightx2v_strength})")
        logger.info(f"  🎨 PUSA: {use_pusa} (strength: {pusa_strength})")
        logger.info(f"  🔄 Interpolation: {enable_interpolation} (factor: {interpolation_factor})")
        
        # ================ SYSTEM VERIFICATION ================
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI modules not available")
        
        with torch.inference_mode():
            # ================ NODE INITIALIZATION ================
            logger.info("🔧 Initializing ComfyUI nodes...")
            
            # Core nodes
            unet_loader = UnetLoaderGGUF()
            pathch_sage_attention = PathchSageAttentionKJ()
            wan_video_nag = WanVideoNAG()
            teacache = WanVideoTeaCacheKJ()
            model_sampling = ModelSamplingSD3()
            clip_loader = CLIPLoader()
            clip_encode_positive = CLIPTextEncode()
            clip_encode_negative = CLIPTextEncode()
            vae_loader = VAELoader()
            clip_vision_loader = CLIPVisionLoader()
            clip_vision_encode = CLIPVisionEncode()
            wan_image_to_video = WanImageToVideo()
            ksampler = KSamplerAdvanced()
            vae_decode = VAEDecode()
            
            # LoRA loaders
            pAssLora = LoraLoaderModelOnly()
            load_lora_node = LoraLoaderModelOnly()
            load_lora2_node = LoraLoaderModelOnly()
            load_lora3_node = LoraLoaderModelOnly()
            load_lightx2v_lora = LoraLoaderModelOnly()
            load_pusa_lora = LoraLoaderModelOnly()
            
            logger.info("✅ All nodes initialized successfully")
            
            # ================ TEXT ENCODING ================
            logger.info("📝 Loading text encoder...")
            clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            
            # Process prompts với prompt assist
            final_positive_prompt = positive_prompt
            if prompt_assist != "none":
                final_positive_prompt = f"{positive_prompt} {prompt_assist}."
                logger.info(f"🎭 Enhanced prompt: {final_positive_prompt}")
            
            positive = clip_encode_positive.encode(clip, final_positive_prompt)[0]
            negative = clip_encode_negative.encode(clip, negative_prompt)[0]
            
            del clip
            clear_memory()
            
            # ================ IMAGE PROCESSING ================
            logger.info("🖼️ Processing image với enhanced aspect control...")
            loaded_image, final_width, final_height = process_image_with_aspect_control(
                image_path=image_path,
                target_width=width,
                target_height=height,
                aspect_mode=aspect_mode,
                auto_720p=auto_720p
            )
            
            # Update dimensions
            width, height = final_width, final_height
            
            # ================ CLIP VISION (OPTIONAL) ================
            clip_vision_output = None
            if use_clip_vision:
                logger.info("👁️ Processing with CLIP Vision...")
                clip_vision = clip_vision_loader.load_clip("clip_vision_h.safetensors")[0]
                clip_vision_output = clip_vision_encode.encode(clip_vision, loaded_image, "none")[0]
                del clip_vision
                clear_memory()
            
            # ================ VAE LOADING ================
            logger.info("🎨 Loading VAE...")
            vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            
            # ================ IMAGE TO VIDEO ENCODING ================
            logger.info("🔄 Encoding image to video latent...")
            positive_out, negative_out, latent = wan_image_to_video.encode(
                positive, negative, vae, width, height, frames, 1, loaded_image, clip_vision_output
            )
            
            # ================ STAGE 1: HIGH NOISE MODEL ================
            logger.info("🎯 STAGE 1: Loading high noise model...")
            model = unet_loader.load_unet("wan2.2_i2v_high_noise_14B_Q6_K.gguf")[0]
            
            # Apply NAG if enabled
            if use_nag:
                logger.info(f"🎯 Applying NAG (strength: {nag_strength})...")
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]
            
            # Apply flow shift
            if enable_flow_shift:
                logger.info(f"🌊 Applying flow shift: {flow_shift}")
                model = model_sampling.patch(model, flow_shift)[0]
            
            # Apply prompt assist LoRAs
            if prompt_assist != "none":
                lora_mapping = {
                    "walking to viewers": "walking to viewers_Wan.safetensors",
                    "walking from behind": "walking_from_behind.safetensors",
                    "b3ll13-d8nc3r": "b3ll13-d8nc3r.safetensors"
                }
                
                if prompt_assist in lora_mapping:
                    lora_file = lora_mapping[prompt_assist]
                    logger.info(f"🎭 Loading prompt assist LoRA: {lora_file}")
                    model = pAssLora.load_lora_model_only(model, lora_file, 1.0)[0]
            
            # Download và apply custom LoRAs
            custom_lora_paths = []
            
            if use_lora and lora_url:
                logger.info("🎨 Processing custom LoRA 1...")
                lora_path = download_lora_dynamic(lora_url, civitai_token)
                if lora_path:
                    model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), lora_strength)[0]
                    custom_lora_paths.append((lora_path, lora_strength))
                    logger.info(f"✅ LoRA 1 applied: {os.path.basename(lora_path)}")
            
            if use_lora2 and lora2_url:
                logger.info("🎨 Processing custom LoRA 2...")
                lora2_path = download_lora_dynamic(lora2_url, civitai_token)
                if lora2_path:
                    model = load_lora2_node.load_lora_model_only(model, os.path.basename(lora2_path), lora2_strength)[0]
                    custom_lora_paths.append((lora2_path, lora2_strength))
                    logger.info(f"✅ LoRA 2 applied: {os.path.basename(lora2_path)}")
            
            if use_lora3 and lora3_url:
                logger.info("🎨 Processing custom LoRA 3...")
                lora3_path = download_lora_dynamic(lora3_url, civitai_token)
                if lora3_path:
                    model = load_lora3_node.load_lora_model_only(model, os.path.basename(lora3_path), lora3_strength)[0]
                    custom_lora_paths.append((lora3_path, lora3_strength))
                    logger.info(f"✅ LoRA 3 applied: {os.path.basename(lora3_path)}")
            
            # Apply LightX2V LoRA
            if use_lightx2v:
                logger.info(f"⚡ Loading LightX2V LoRA rank {lightx2v_rank} (strength: {lightx2v_strength})...")
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path and os.path.exists(lightx2v_lora_path):
                    model = load_lightx2v_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), lightx2v_strength
                    )[0]
                    logger.info(f"✅ LightX2V LoRA applied successfully")
                else:
                    logger.warning(f"⚠️ LightX2V LoRA not found: {lightx2v_lora_path}")
            
            # Apply optimizations
            if use_sage_attention:
                logger.info("🧠 Applying Sage Attention...")
                try:
                    model = pathch_sage_attention.patch(model, "auto")[0]
                    logger.info("✅ Sage Attention applied")
                except Exception as e:
                    logger.warning(f"⚠️ Sage Attention failed: {e}")
            
            if rel_l1_thresh > 0:
                logger.info(f"🫖 Applying TeaCache: threshold={rel_l1_thresh}")
                try:
                    model = teacache.patch_teacache(model, rel_l1_thresh, start_percent, end_percent, "main_device", "14B")[0]
                    logger.info("✅ TeaCache applied")
                except Exception as e:
                    logger.warning(f"⚠️ TeaCache failed: {e}")
            
            # Sample với high noise model
            logger.info(f"🎬 Sampling with high noise model (steps: {steps}, end_step: {high_noise_steps})...")
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
            clear_memory()
            
            # ================ STAGE 2: LOW NOISE MODEL ================
            logger.info("🎯 STAGE 2: Loading low noise model...")
            model = unet_loader.load_unet("wan2.2_i2v_low_noise_14B_Q6_K.gguf")[0]
            
            # Reapply optimizations
            if use_nag:
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]
            
            if enable_flow_shift2:
                logger.info(f"🌊 Applying flow shift 2: {flow_shift2}")
                model = model_sampling.patch(model, flow_shift2)[0]
            
            # Reapply LoRAs
            if prompt_assist != "none" and prompt_assist in lora_mapping:
                model = pAssLora.load_lora_model_only(model, lora_mapping[prompt_assist], 1.0)[0]
            
            # Reapply custom LoRAs
            for lora_path, strength in custom_lora_paths:
                model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), strength)[0]
            
            # Apply PUSA LoRA
            if use_pusa:
                logger.info(f"🎭 Loading PUSA LoRA (strength: {pusa_strength})...")
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path and os.path.exists(lightx2v_lora_path):
                    model = load_pusa_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), pusa_strength
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
                    model = teacache.patch_teacache(model, rel_l1_thresh, start_percent, end_percent, "main_device", "14B")[0]
                except Exception as e:
                    logger.warning(f"⚠️ TeaCache (stage 2) failed: {e}")
            
            # Sample với low noise model
            logger.info(f"🎬 Sampling with low noise model (start_step: {high_noise_steps})...")
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
            clear_memory()
            
            # ================ DECODING ================
            logger.info("🎨 Decoding latents to video frames...")
            decoded = vae_decode.decode(vae, sampled)[0]
            
            del vae
            clear_memory()
            
            # ================ VIDEO SAVING ================
            logger.info("💾 Saving video với production method...")
            output_path = f"/app/ComfyUI/output/wan22_production_{uuid.uuid4().hex[:8]}.mp4"
            
            final_output_path = save_video_optimized(decoded, output_path, fps)
            
            # ================ FRAME INTERPOLATION ================
            if enable_interpolation and interpolation_factor > 1:
                logger.info(f"🔄 Applying PRODUCTION RIFE interpolation (factor: {interpolation_factor})...")
                
                pre_interp_size = os.path.getsize(final_output_path) / (1024 * 1024)
                logger.info(f"📊 Pre-interpolation: {pre_interp_size:.1f}MB")
                
                # Apply RIFE với production method
                interpolated_path = apply_rife_interpolation_production(final_output_path, interpolation_factor)
                
                if interpolated_path != final_output_path and os.path.exists(interpolated_path):
                    post_interp_size = os.path.getsize(interpolated_path) / (1024 * 1024)
                    logger.info(f"📊 Post-interpolation: {post_interp_size:.1f}MB")
                    logger.info(f"✅ Frame interpolation successful!")
                    return interpolated_path
                else:
                    logger.warning("⚠️ Frame interpolation failed, using original video")
                    return final_output_path
            
            return final_output_path
            
    except Exception as e:
        logger.error(f"❌ Video generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        clear_memory()

# ================== MINIO UPLOAD ==================

def upload_to_minio(local_path: str, object_name: str) -> str:
    """📤 Upload file to MinIO storage với enhanced error handling"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading to MinIO: {object_name} ({file_size_mb:.1f}MB)")
        
        # Upload với progress tracking
        start_time = time.time()
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        upload_time = time.time() - start_time
        
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Upload completed: {file_url}")
        logger.info(f"📊 Upload stats: {file_size_mb:.1f}MB in {upload_time:.1f}s")
        
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

# ================== INPUT VALIDATION ==================

def validate_input_parameters(job_input: dict) -> tuple[bool, str]:
    """🔍 COMPREHENSIVE input parameter validation"""
    try:
        # Required parameters
        required_params = ["image_url", "positive_prompt"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        # Validate image URL accessibility
        image_url = job_input["image_url"]
        try:
            response = requests.head(image_url, timeout=10)
            if response.status_code not in [200, 301, 302]:
                return False, f"Image URL not accessible: {response.status_code}"
        except Exception as e:
            return False, f"Image URL validation failed: {str(e)}"
        
        # Validate dimensions
        width = job_input.get("width", 720)
        height = job_input.get("height", 1280)
        if not (256 <= width <= 1024 and 256 <= height <= 1536):
            return False, "Width must be 256-1024, height must be 256-1536"
        
        # Validate frames
        frames = job_input.get("frames", 65)
        if not (1 <= frames <= 120):
            return False, "Frames must be between 1 and 120"
        
        # Validate aspect mode
        aspect_mode = job_input.get("aspect_mode", "preserve")
        if aspect_mode not in ["preserve", "crop", "stretch"]:
            return False, "aspect_mode must be one of: preserve, crop, stretch"
        
        # Validate LightX2V rank
        lightx2v_rank = job_input.get("lightx2v_rank", "32")
        if lightx2v_rank not in ["32", "64", "128"]:
            return False, "LightX2V rank must be one of: 32, 64, 128"
        
        # Validate sampler
        sampler_name = job_input.get("sampler_name", "euler")
        valid_samplers = ["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m"]
        if sampler_name not in valid_samplers:
            return False, f"Invalid sampler. Must be one of: {', '.join(valid_samplers)}"
        
        # Validate numerical ranges
        cfg_scale = job_input.get("cfg_scale", 1.0)
        if not (0.1 <= cfg_scale <= 10.0):
            return False, "CFG scale must be between 0.1 and 10.0"
        
        steps = job_input.get("steps", 6)
        if not (1 <= steps <= 30):
            return False, "Steps must be between 1 and 30"
        
        high_noise_steps = job_input.get("high_noise_steps", 3)
        if not (1 <= high_noise_steps < steps):
            return False, f"High noise steps must be between 1 and {steps-1}"
        
        interpolation_factor = job_input.get("interpolation_factor", 2)
        if not (2 <= interpolation_factor <= 8):
            return False, "Interpolation factor must be between 2 and 8"
        
        # Validate prompt assist
        prompt_assist = job_input.get("prompt_assist", "none")
        valid_assists = ["none", "walking to viewers", "walking from behind", "b3ll13-d8nc3r"]
        if prompt_assist not in valid_assists:
            return False, f"Prompt assist must be one of: {', '.join(valid_assists)}"
        
        return True, "All parameters valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

# ================== MAIN HANDLER ==================

def handler(job):
    """
    🚀 PRODUCTION GRADE: Main RunPod handler cho WAN2.2 LightX2V Q6_K
    ✅ All features, fixes, optimizations, và stability improvements
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Input validation
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            return {
                "error": validation_message,
                "status": "failed",
                "job_id": job_id,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        
        # Extract parameters
        image_url = job_input["image_url"]
        positive_prompt = job_input["positive_prompt"]
        
        # Build complete parameter set
        parameters = {
            # Core parameters
            "positive_prompt": positive_prompt,
            "negative_prompt": job_input.get("negative_prompt", "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
            
            # Video settings
            "width": job_input.get("width", 720),
            "height": job_input.get("height", 1280),
            "seed": job_input.get("seed", 0),
            "steps": job_input.get("steps", 6),
            "high_noise_steps": job_input.get("high_noise_steps", 3),
            "cfg_scale": job_input.get("cfg_scale", 1.0),
            "sampler_name": job_input.get("sampler_name", "euler"),
            "scheduler": job_input.get("scheduler", "simple"),
            "frames": job_input.get("frames", 65),
            "fps": job_input.get("fps", 16),
            
            # Enhanced aspect control
            "aspect_mode": job_input.get("aspect_mode", "preserve"),
            "auto_720p": job_input.get("auto_720p", False),
            
            # Advanced features
            "prompt_assist": job_input.get("prompt_assist", "none"),
            
            # LoRA settings
            "use_lora": job_input.get("use_lora", False),
            "lora_url": job_input.get("lora_url", None),
            "lora_strength": job_input.get("lora_strength", 1.0),
            "civitai_token": job_input.get("civitai_token", None),
            "use_lora2": job_input.get("use_lora2", False),
            "lora2_url": job_input.get("lora2_url", None),
            "lora2_strength": job_input.get("lora2_strength", 1.0),
            "use_lora3": job_input.get("use_lora3", False),
            "lora3_url": job_input.get("lora3_url", None),
            "lora3_strength": job_input.get("lora3_strength", 1.0),
            
            # LightX2V & PUSA
            "use_lightx2v": job_input.get("use_lightx2v", True),
            "lightx2v_rank": job_input.get("lightx2v_rank", "32"),
            "lightx2v_strength": job_input.get("lightx2v_strength", 3.0),
            "use_pusa": job_input.get("use_pusa", True),
            "pusa_strength": job_input.get("pusa_strength", 1.5),
            
            # Optimization settings
            "use_sage_attention": job_input.get("use_sage_attention", True),
            "rel_l1_thresh": job_input.get("rel_l1_thresh", 0.0),
            "start_percent": job_input.get("start_percent", 0.2),
            "end_percent": job_input.get("end_percent", 1.0),
            
            # Flow shift
            "enable_flow_shift": job_input.get("enable_flow_shift", True),
            "flow_shift": job_input.get("flow_shift", 8.0),
            "enable_flow_shift2": job_input.get("enable_flow_shift2", True),
            "flow_shift2": job_input.get("flow_shift2", 8.0),
            
            # Advanced optimizations
            "use_nag": job_input.get("use_nag", False),
            "nag_strength": job_input.get("nag_strength", 11.0),
            "nag_scale1": job_input.get("nag_scale1", 0.25),
            "nag_scale2": job_input.get("nag_scale2", 2.5),
            "use_clip_vision": job_input.get("use_clip_vision", False),
            
            # Frame interpolation
            "enable_interpolation": job_input.get("enable_interpolation", False),
            "interpolation_factor": job_input.get("interpolation_factor", 2)
        }
        
        # Log job start
        logger.info(f"🚀 Job {job_id}: WAN2.2 PRODUCTION Generation Started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"📝 Prompt: {positive_prompt[:100]}...")
        logger.info(f"⚙️ Settings: {parameters['width']}x{parameters['height']}, {parameters['frames']} frames")
        logger.info(f"🎨 Features: aspect_mode={parameters['aspect_mode']}, interpolation={parameters['enable_interpolation']}")
        
        # Verify models
        models_ok, missing_models = verify_models()
        if not models_ok:
            return {
                "error": "Required models are missing",
                "missing_models": missing_models,
                "status": "failed",
                "job_id": job_id
            }
        
        # Process with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download input image
            image_path = os.path.join(temp_dir, "input_image.jpg")
            logger.info("📥 Downloading input image...")
            
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
                return {
                    "error": f"Failed to download image: {str(e)}",
                    "status": "failed",
                    "job_id": job_id
                }
            
            # Generate video
            logger.info("🎬 Starting PRODUCTION video generation...")
            generation_start = time.time()
            
            output_path = generate_video_wan22_complete(
                image_path=image_path,
                **parameters
            )
            
            generation_time = time.time() - generation_start
            
            if not output_path or not os.path.exists(output_path):
                return {
                    "error": "Video generation failed - no output produced",
                    "status": "failed",
                    "job_id": job_id,
                    "processing_time_seconds": round(time.time() - start_time, 2)
                }
            
            # Upload result
            logger.info("📤 Uploading result to storage...")
            output_filename = f"wan22_production_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            
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
            duration_seconds = parameters["frames"] / parameters["fps"]
            
            # Adjust duration for interpolation
            if parameters["enable_interpolation"] and "_rife_" in os.path.basename(output_path):
                # RIFE doesn't change duration, just increases smoothness
                pass  # Duration stays the same
            
            actual_seed = parameters["seed"] if parameters["seed"] != 0 else "auto-generated"
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.1f}s (generation: {generation_time:.1f}s)")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB, {duration_seconds:.1f}s duration")
            
            # Return comprehensive results
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(total_time, 2),
                "generation_time_seconds": round(generation_time, 2),
                
                "video_info": {
                    "width": parameters["width"],
                    "height": parameters["height"],
                    "frames": parameters["frames"],
                    "fps": parameters["fps"],
                    "duration_seconds": round(duration_seconds, 2),
                    "file_size_mb": round(file_size_mb, 2),
                    "interpolated": parameters["enable_interpolation"] and "_rife_" in os.path.basename(output_path),
                    "interpolation_factor": parameters["interpolation_factor"] if parameters["enable_interpolation"] else 1,
                    "aspect_mode": parameters["aspect_mode"],
                    "auto_720p": parameters["auto_720p"]
                },
                
                "generation_params": {
                    "positive_prompt": parameters["positive_prompt"],
                    "negative_prompt": parameters["negative_prompt"][:100] + "..." if len(parameters["negative_prompt"]) > 100 else parameters["negative_prompt"],
                    "steps": parameters["steps"],
                    "high_noise_steps": parameters["high_noise_steps"],
                    "cfg_scale": parameters["cfg_scale"],
                    "seed": actual_seed,
                    "sampler_name": parameters["sampler_name"],
                    "scheduler": parameters["scheduler"],
                    "prompt_assist": parameters["prompt_assist"],
                    
                    "lightx2v_config": {
                        "enabled": parameters["use_lightx2v"],
                        "rank": parameters["lightx2v_rank"],
                        "strength": parameters["lightx2v_strength"]
                    },
                    
                    "pusa_config": {
                        "enabled": parameters["use_pusa"],
                        "strength": parameters["pusa_strength"]
                    },
                    
                    "lora_config": {
                        "lora1": {
                            "enabled": parameters["use_lora"],
                            "url": parameters["lora_url"],
                            "strength": parameters["lora_strength"]
                        },
                        "lora2": {
                            "enabled": parameters["use_lora2"],
                            "url": parameters["lora2_url"],
                            "strength": parameters["lora2_strength"]
                        },
                        "lora3": {
                            "enabled": parameters["use_lora3"],
                            "url": parameters["lora3_url"],
                            "strength": parameters["lora3_strength"]
                        }
                    },
                    
                    "optimizations": {
                        "sage_attention": parameters["use_sage_attention"],
                        "teacache_threshold": parameters["rel_l1_thresh"],
                        "flow_shift": parameters["enable_flow_shift"],
                        "nag_enabled": parameters["use_nag"],
                        "clip_vision": parameters["use_clip_vision"]
                    },
                    
                    "interpolation": {
                        "enabled": parameters["enable_interpolation"],
                        "factor": parameters["interpolation_factor"]
                    },
                    
                    "aspect_control": {
                        "mode": parameters["aspect_mode"],
                        "auto_720p": parameters["auto_720p"]
                    },
                    
                    "model_quantization": "Q6_K",
                    "workflow_version": "PRODUCTION_COMPLETE_v2.0"
                },
                
                "status": "completed"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": error_msg,
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id
        }
        
    finally:
        clear_memory()

# ================== HEALTH CHECK ==================

def health_check():
    """🏥 Comprehensive system health check"""
    try:
        health_issues = []
        
        # Check CUDA
        if not torch.cuda.is_available():
            health_issues.append("CUDA not available")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 10:  # Less than 10GB
                health_issues.append(f"Low GPU memory: {gpu_memory:.1f}GB")
        
        # Check models
        models_ok, missing = verify_models()
        if not models_ok:
            health_issues.append(f"Missing {len(missing)} models")
        
        # Check ComfyUI
        if not COMFYUI_AVAILABLE:
            health_issues.append("ComfyUI not available")
        
        # Check MinIO
        if not minio_client:
            health_issues.append("MinIO not available")
        
        # Check directories
        required_dirs = ["/app/ComfyUI", "/app/Practical-RIFE"]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                health_issues.append(f"Missing directory: {dir_path}")
        
        if health_issues:
            return False, f"Health issues: {'; '.join(health_issues)}"
        else:
            return True, "All systems operational - PRODUCTION READY"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

# ================== MAIN ENTRY POINT ==================

if __name__ == "__main__":
    logger.info("🚀 Starting WAN2.2 LightX2V PRODUCTION Serverless Worker...")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # Comprehensive startup health check
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            logger.error("💥 System not ready for production use!")
            sys.exit(1)
        
        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎬 PRODUCTION READY - All systems optimized!")
        
        # Feature summary
        logger.info("📋 Available Features:")
        logger.info("  ✅ Enhanced Aspect Ratio Control (preserve/crop/stretch)")
        logger.info("  ✅ RIFE Frame Interpolation (FIXED)")
        logger.info("  ✅ LightX2V LoRA Support (rank 32/64/128)")
        logger.info("  ✅ Custom LoRA Support (HuggingFace/CivitAI)")
        logger.info("  ✅ Advanced Optimizations (Sage Attention, TeaCache)")
        logger.info("  ✅ Comprehensive Error Handling")
        logger.info("  ✅ Production Grade Monitoring")
        
        # Start RunPod serverless worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
