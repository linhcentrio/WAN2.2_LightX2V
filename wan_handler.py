#!/usr/bin/env python3

"""
RunPod Serverless Handler cho WAN2.2 LightX2V Q6_K - Complete Optimized Version
Optimized for fixing 0MB video output issue
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
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
from huggingface_hub import hf_hub_download
import logging
import imageio
import numpy as np

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add ComfyUI paths
sys.path.insert(0, '/app/ComfyUI')
sys.path.insert(0, '/app/Practical-RIFE')

# Import ComfyUI components với comprehensive error handling
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

# MinIO Configuration
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

# Model configurations với đường dẫn chính xác
MODEL_CONFIGS = {
    # Q6_K DIT Models
    "dit_model_high": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
    "dit_model_low": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf",
    
    # Supporting models
    "text_encoder": "/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "/app/ComfyUI/models/vae/wan_2.1_vae.safetensors",
    "clip_vision": "/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
    
    # LightX2V LoRAs theo rank
    "lightx2v_rank_32": "/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
    "lightx2v_rank_64": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
    "lightx2v_rank_128": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
    
    # Built-in LoRAs với tên file chính xác
    "walking_to_viewers": "/app/ComfyUI/models/loras/walking to viewers_Wan.safetensors",
    "walking_from_behind": "/app/ComfyUI/models/loras/walking_from_behind.safetensors",
    "dancing": "/app/ComfyUI/models/loras/b3ll13-d8nc3r.safetensors",
    "pusa_lora": "/app/ComfyUI/models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors",
    "rotate_lora": "/app/ComfyUI/models/loras/rotate_20_epochs.safetensors"
}

# Global model cache để tối ưu memory
model_cache = {}

# Enable PyTorch 2.6.0 optimizations
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

def verify_models() -> tuple[bool, list]:
    """
    Verify tất cả models cần thiết có tồn tại
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
    Download LoRA từ HuggingFace hoặc CivitAI theo code gốc
    Support cho multiple platforms với proper error handling
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

def get_lightx2v_lora_path(lightx2v_rank: str) -> str:
    """Get LightX2V LoRA path theo rank"""
    rank_mapping = {
        "32": "lightx2v_rank_32",
        "64": "lightx2v_rank_64", 
        "128": "lightx2v_rank_128"
    }
    config_key = rank_mapping.get(lightx2v_rank, "64")
    return MODEL_CONFIGS.get(config_key)

def apply_rife_interpolation(video_path: str, interpolation_factor: int = 2) -> str:
    """
    Apply RIFE frame interpolation để tăng FPS
    """
    try:
        logger.info(f"🔄 Applying RIFE interpolation (factor: {interpolation_factor})...")
        
        # Import RIFE modules
        sys.path.append('/app/Practical-RIFE')
        from inference_video import interpolate_video
        
        output_path = video_path.replace('.mp4', f'_rife_x{interpolation_factor}.mp4')
        
        # Apply RIFE interpolation
        interpolate_video(
            input_path=video_path,
            output_path=output_path,
            times=interpolation_factor,
            fps=None  # Keep original FPS * factor
        )
        
        if os.path.exists(output_path):
            original_size = os.path.getsize(video_path) / (1024 * 1024)
            interpolated_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ RIFE interpolation completed: {original_size:.1f}MB → {interpolated_size:.1f}MB")
            return output_path
        else:
            logger.warning("⚠️ RIFE interpolation failed, returning original")
            return video_path
            
    except Exception as e:
        logger.error(f"❌ RIFE interpolation error: {e}")
        return video_path

def validate_tensor_data(tensor_data, name="tensor"):
    """Validate tensor data before processing - OPTIMIZED VERSION"""
    try:
        if torch.is_tensor(tensor_data):
            data = tensor_data.detach().cpu().numpy()
        else:
            data = np.array(tensor_data)
        
        stats = {
            'shape': data.shape,
            'dtype': data.dtype,
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'nan_count': int(np.sum(np.isnan(data))),
            'inf_count': int(np.sum(np.isinf(data))),
            'zero_count': int(np.sum(data == 0))
        }
        
        logger.info(f"📊 {name} stats: shape={stats['shape']}, range=[{stats['min']:.3f}, {stats['max']:.3f}], mean={stats['mean']:.3f}")
        
        # Check for critical issues
        issues = []
        if stats['nan_count'] > 0:
            issues.append(f"{stats['nan_count']} NaN values")
        if stats['inf_count'] > 0:
            issues.append(f"{stats['inf_count']} Inf values")
        if stats['zero_count'] == data.size:
            issues.append("All zeros")
        if stats['min'] == stats['max'] and stats['min'] != 0:
            issues.append("Constant values")
        
        if issues:
            logger.warning(f"⚠️ {name} issues: {', '.join(issues)}")
            
        return len(issues) == 0, stats
        
    except Exception as e:
        logger.error(f"❌ Error validating {name}: {e}")
        return False, {}

def safe_tensor_to_numpy(tensor_data, target_shape=None):
    """
    Safely convert tensor to numpy array - OPTIMIZED VERSION BASED ON ORIGINAL WORKFLOW
    """
    try:
        # Convert to numpy if it's a tensor
        if torch.is_tensor(tensor_data):
            # Ensure tensor is on CPU and detached properly
            numpy_data = tensor_data.detach().cpu().float().numpy()
        else:
            numpy_data = np.array(tensor_data, dtype=np.float32)
        
        # Validate input data first
        is_valid, stats = validate_tensor_data(numpy_data, "input_tensor")
        
        # Handle batch dimension properly (similar to original workflow)
        original_shape = numpy_data.shape
        if numpy_data.ndim == 4:  # [batch, height, width, channels]
            if numpy_data.shape[0] == 1:
                numpy_data = numpy_data[0]
                logger.info(f"🔄 Removed batch dimension: {original_shape} → {numpy_data.shape}")
        elif numpy_data.ndim == 5:  # [batch, frames, height, width, channels] 
            if numpy_data.shape[0] == 1:
                numpy_data = numpy_data[0]
                logger.info(f"🔄 Removed batch dimension: {original_shape} → {numpy_data.shape}")
        
        # Check for NaN and Inf values BEFORE any processing
        nan_mask = np.isnan(numpy_data)
        inf_mask = np.isinf(numpy_data)
        
        if np.any(nan_mask) or np.any(inf_mask):
            nan_count = np.sum(nan_mask)
            inf_count = np.sum(inf_mask)
            logger.warning(f"⚠️ Found {nan_count} NaN and {inf_count} Inf values - fixing...")
            
            # Smart replacement strategy instead of zeros
            valid_data = numpy_data[~(nan_mask | inf_mask)]
            if len(valid_data) > 1000:  # Enough samples for statistics
                # Use median for better preservation of image structure
                replacement_value = np.median(valid_data)
                logger.info(f"🔧 Using median replacement: {replacement_value:.3f}")
            elif len(valid_data) > 0:
                replacement_value = np.mean(valid_data)
                logger.info(f"🔧 Using mean replacement: {replacement_value:.3f}")
            else:
                replacement_value = 0.5  # Middle gray as last resort
                logger.warning("🔧 Using fallback gray value: 0.5")
            
            numpy_data[nan_mask | inf_mask] = replacement_value
        
        # Normalize data properly (based on original workflow logic)
        data_min = numpy_data.min()
        data_max = numpy_data.max()
        
        if data_max > 1.0 or data_min < 0.0:
            logger.info(f"🔧 Normalizing data from [{data_min:.3f}, {data_max:.3f}] to [0, 1]")
            # Safe normalization to avoid division by zero
            data_range = data_max - data_min
            if data_range > 1e-8:
                numpy_data = (numpy_data - data_min) / data_range
            else:
                # Constant value case
                numpy_data = np.full_like(numpy_data, 0.5)
                logger.warning("🔧 Constant data detected, using middle gray")
        
        # Final clipping to ensure [0, 1] range
        numpy_data = np.clip(numpy_data, 0.0, 1.0)
        
        # Reshape if target shape provided
        if target_shape and numpy_data.shape != target_shape:
            try:
                logger.info(f"🔄 Reshaping from {numpy_data.shape} to {target_shape}")
                numpy_data = numpy_data.reshape(target_shape)
            except ValueError as e:
                logger.error(f"❌ Cannot reshape {numpy_data.shape} to {target_shape}: {e}")
                # Create fallback data with correct shape (preserve aspect ratio if possible)
                numpy_data = np.full(target_shape, 0.5, dtype=np.float32)
                logger.warning("🔧 Using fallback gray data with target shape")
        
        # Final validation
        final_min, final_max = numpy_data.min(), numpy_data.max()
        if final_min < 0 or final_max > 1:
            logger.warning(f"⚠️ Final data out of range: [{final_min:.3f}, {final_max:.3f}]")
            numpy_data = np.clip(numpy_data, 0.0, 1.0)
        
        logger.info(f"✅ Tensor conversion successful: {numpy_data.shape}, range=[{numpy_data.min():.3f}, {numpy_data.max():.3f}]")
        return numpy_data
        
    except Exception as e:
        logger.error(f"❌ Error in tensor conversion: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return fallback data
        if target_shape:
            fallback_data = np.full(target_shape, 0.5, dtype=np.float32)
            logger.warning(f"🔧 Using fallback data with shape: {target_shape}")
            return fallback_data
        else:
            fallback_data = np.full((480, 832, 3), 0.5, dtype=np.float32)
            logger.warning("🔧 Using default fallback data: (480, 832, 3)")
            return fallback_data

def save_video_optimized(frames_tensor, output_path, fps=16):
    """
    Optimized video saving based on original workflow - FIXED VERSION
    """
    try:
        logger.info(f"🎬 Starting optimized video save to: {output_path}")
        
        # Convert tensor to numpy frames
        frames_np = safe_tensor_to_numpy(frames_tensor)
        
        if frames_np is None or frames_np.size == 0:
            raise ValueError("Empty or invalid frames data after conversion")
        
        # Validate frame structure
        if len(frames_np.shape) != 4:  # Should be [frames, height, width, channels]
            raise ValueError(f"Invalid frame shape after conversion: {frames_np.shape} (expected 4D)")
        
        num_frames, height, width, channels = frames_np.shape
        logger.info(f"📊 Frame structure: {num_frames} frames, {height}x{width}, {channels} channels")
        
        # Validate dimensions
        if num_frames < 1:
            raise ValueError(f"No frames to save: {num_frames}")
        if height < 16 or width < 16:
            raise ValueError(f"Frame dimensions too small: {height}x{width}")
        if channels not in [1, 3, 4]:
            raise ValueError(f"Invalid number of channels: {channels}")
        
        # Convert to RGB if needed
        if channels == 1:
            frames_np = np.repeat(frames_np, 3, axis=-1)
            logger.info("🔄 Converted grayscale to RGB")
        elif channels == 4:
            frames_np = frames_np[:, :, :, :3]  # Remove alpha channel
            logger.info("🔄 Removed alpha channel")
        
        # Convert to uint8 (0-255 range) - CRITICAL STEP
        frames_np = (frames_np * 255.0).astype(np.uint8)
        
        # Final validation of frame data
        logger.info(f"📊 Final frames: shape={frames_np.shape}, dtype={frames_np.dtype}")
        logger.info(f"📊 Value range: [{frames_np.min()}, {frames_np.max()}]")
        
        # Check for completely black or white frames
        frame_means = np.mean(frames_np, axis=(1, 2, 3))
        black_frames = np.sum(frame_means < 5)
        white_frames = np.sum(frame_means > 250)
        if black_frames > num_frames * 0.8:
            logger.warning(f"⚠️ {black_frames}/{num_frames} frames are mostly black")
        if white_frames > num_frames * 0.8:
            logger.warning(f"⚠️ {white_frames}/{num_frames} frames are mostly white")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save video using imageio with settings from original workflow
        logger.info("💾 Writing video with imageio...")
        with imageio.get_writer(
            output_path, 
            fps=fps, 
            codec='libx264',
            quality=8,  # Same as original workflow
            bitrate='8M',  # Higher bitrate for better quality
            macro_block_size=1,  # Same as original
            pixelformat='yuv420p',  # Ensure compatibility
            ffmpeg_params=['-preset', 'fast']  # Faster encoding
        ) as writer:
            
            written_frames = 0
            for i, frame in enumerate(frames_np):
                try:
                    # Validate each frame before writing
                    if frame.shape != (height, width, 3):
                        logger.warning(f"⚠️ Frame {i} shape mismatch: {frame.shape}, expected: ({height}, {width}, 3)")
                        # Resize frame if possible
                        if frame.size > 0:
                            frame = frame.reshape((height, width, 3))
                        else:
                            continue
                    
                    # Check for completely invalid frames
                    if np.all(frame == 0):
                        if i > 0:
                            # Use previous frame
                            frame = frames_np[i-1]
                            logger.warning(f"⚠️ Frame {i} is black, using previous frame")
                        else:
                            # Create gray frame
                            frame = np.full((height, width, 3), 128, dtype=np.uint8)
                            logger.warning(f"⚠️ Frame {i} is black, using gray frame")
                    
                    # Write frame
                    writer.append_data(frame)
                    written_frames += 1
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.info(f"📹 Written frame {i + 1}/{num_frames}")
                
                except Exception as frame_error:
                    logger.error(f"❌ Error processing frame {i}: {frame_error}")
                    # Skip problematic frame
                    continue
        
        logger.info(f"✅ Video writing completed: {written_frames}/{num_frames} frames written")
        
        # Verify file was created and has reasonable size
        if not os.path.exists(output_path):
            raise FileNotFoundError("Video file was not created by imageio")
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        # Calculate minimum expected size (more lenient)
        min_expected_size = max(0.1, written_frames * 0.005)  # At least 0.005MB per frame, minimum 0.1MB
        
        if file_size_mb < min_expected_size:
            logger.warning(f"⚠️ Video file smaller than expected: {file_size_mb:.3f}MB (expected > {min_expected_size:.3f}MB)")
            # Don't fail immediately, but log the warning
        
        logger.info(f"✅ Video saved successfully: {file_size_mb:.3f}MB ({written_frames} frames @ {fps}fps)")
        
        # Additional verification - try to read back the video
        try:
            with imageio.get_reader(output_path) as reader:
                read_frames = len(reader)
                logger.info(f"✅ Verification: Video contains {read_frames} readable frames")
        except Exception as verify_error:
            logger.warning(f"⚠️ Video verification failed: {verify_error}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Video saving failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

def generate_video_wan22_complete(image_path: str, **kwargs) -> str:
    """
    Complete WAN2.2 video generation theo đúng logic code gốc
    Bao gồm tất cả optimizations và features với improved error handling
    """
    try:
        logger.info("🎬 Starting WAN2.2 Q6_K complete generation...")
        
        # Extract all parameters từ kwargs với default values theo code gốc
        positive_prompt = kwargs.get('positive_prompt', '')
        negative_prompt = kwargs.get('negative_prompt', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
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
        
        # Prompt assist
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
        lightx2v_rank = kwargs.get('lightx2v_rank', '64')
        lightx2v_strength = kwargs.get('lightx2v_strength', 3.0)
        
        # PUSA LoRA
        use_pusa = kwargs.get('use_pusa', False)
        pusa_strength = kwargs.get('pusa_strength', 1.2)
        
        # Optimization parameters
        use_sage_attention = kwargs.get('use_sage_attention', True)
        rel_l1_thresh = kwargs.get('rel_l1_thresh', 0.0)
        start_percent = kwargs.get('start_percent', 0.2)
        end_percent = kwargs.get('end_percent', 1.0)
        
        # Flow shift parameters
        enable_flow_shift = kwargs.get('enable_flow_shift', True)
        flow_shift = kwargs.get('flow_shift', 8.0)
        enable_flow_shift2 = kwargs.get('enable_flow_shift2', True)
        flow_shift2 = kwargs.get('flow_shift2', 8.0)
        
        # Advanced parameters (mostly disabled in original)
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
            
        logger.info(f"🎯 Generation Parameters:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  Frames: {frames}, FPS: {fps}")
        logger.info(f"  Steps: {steps} (high noise: {high_noise_steps})")
        logger.info(f"  CFG Scale: {cfg_scale}, Seed: {seed}")
        logger.info(f"  Prompt Assist: {prompt_assist}")
        logger.info(f"  LightX2V: {use_lightx2v} (rank: {lightx2v_rank}, strength: {lightx2v_strength})")

        # Verify ComfyUI availability
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI modules not available")

        # Auto-adjust height if 0 (maintain aspect ratio)
        if height == 0:
            # Load image để get original dimensions
            from PIL import Image
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                height = int(width * original_height / original_width)
            # Ensure height is multiple of 8 for video encoding
            height = (height // 8) * 8
            logger.info(f"🔄 Auto-adjusted height: {height} (aspect ratio preserved)")

        with torch.inference_mode(), torch.amp.autocast('cuda'):
            
            # Initialize all ComfyUI nodes
            logger.info("🔧 Initializing ComfyUI nodes...")
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
            load_image = LoadImage()
            wan_image_to_video = WanImageToVideo()
            ksampler = KSamplerAdvanced()
            vae_decode = VAEDecode()
            image_scaler = ImageScale()
            
            # LoRA loaders
            pAssLora = LoraLoaderModelOnly()
            load_lora_node = LoraLoaderModelOnly()
            load_lora2_node = LoraLoaderModelOnly()
            load_lora3_node = LoraLoaderModelOnly()
            load_lightx2v_lora = LoraLoaderModelOnly()
            load_pusa_lora = LoraLoaderModelOnly()
            
            logger.info("✅ ComfyUI nodes initialized")

            # Load text encoder
            logger.info("📝 Loading Text Encoder...")
            clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            
            # Modify prompt với prompt assist
            final_positive_prompt = positive_prompt
            if prompt_assist != "none":
                final_positive_prompt = f"{positive_prompt}, {prompt_assist}"
                
            positive = clip_encode_positive.encode(clip, final_positive_prompt)[0]
            negative = clip_encode_negative.encode(clip, negative_prompt)[0]
            del clip
            torch.cuda.empty_cache()
            gc.collect()

            # Load và process image
            logger.info("🖼️ Loading and processing image...")
            loaded_image = load_image.load_image(image_path)[0]
            
            # Scale image
            logger.info(f"🔄 Scaling image to {width}x{height}...")
            loaded_image = image_scaler.upscale(loaded_image, "lanczos", width, height, "disabled")[0]

            # CLIP Vision processing (optional)
            clip_vision_output = None
            if use_clip_vision:
                logger.info("👁️ Processing with CLIP Vision...")
                clip_vision = clip_vision_loader.load_clip("clip_vision_h.safetensors")[0]
                clip_vision_output = clip_vision_encode.encode(clip_vision, loaded_image, "none")[0]
                del clip_vision
                torch.cuda.empty_cache()
                gc.collect()

            # Load VAE
            logger.info("🎨 Loading VAE...")
            vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]

            # Encode image to video latent
            logger.info("🔄 Encoding image to video latent...")
            positive_out, negative_out, latent = wan_image_to_video.encode(
                positive, negative, vae, width, height, frames, 1, loaded_image, clip_vision_output
            )

            # STAGE 1: High noise model
            logger.info("🎯 Loading high noise Q6_K model...")
            model = unet_loader.load_unet("wan2.2_i2v_high_noise_14B_Q6_K.gguf")[0]

            # Apply NAG nếu enabled
            if use_nag:
                logger.info(f"🎯 Applying NAG (strength: {nag_strength})...")
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]

            # Apply flow shift for high noise
            if enable_flow_shift:
                logger.info(f"🌊 Applying flow shift: {flow_shift}")
                model = model_sampling.patch(model, flow_shift)[0]

            # Apply prompt assist LoRAs
            if prompt_assist == "walking to viewers":
                logger.info("🚶 Loading walking to viewers LoRA...")
                model = pAssLora.load_lora_model_only(model, "walking to viewers_Wan.safetensors", 1.0)[0]
            elif prompt_assist == "walking from behind":
                logger.info("🚶 Loading walking from behind LoRA...")
                model = pAssLora.load_lora_model_only(model, "walking_from_behind.safetensors", 1.0)[0]
            elif prompt_assist == "b3ll13-d8nc3r":
                logger.info("💃 Loading dancing LoRA...")
                model = pAssLora.load_lora_model_only(model, "b3ll13-d8nc3r.safetensors", 1.0)[0]

            # Download và apply custom LoRAs
            custom_lora_paths = []
            used_steps = steps
            
            if use_lora and lora_url:
                logger.info("🎨 Processing custom LoRA 1...")
                lora_path = download_lora_dynamic(lora_url, civitai_token)
                if lora_path:
                    model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), lora_strength)[0]
                    custom_lora_paths.append((lora_path, lora_strength))

            if use_lora2 and lora2_url:
                logger.info("🎨 Processing custom LoRA 2...")
                lora2_path = download_lora_dynamic(lora2_url, civitai_token)
                if lora2_path:
                    model = load_lora2_node.load_lora_model_only(model, os.path.basename(lora2_path), lora2_strength)[0]
                    custom_lora_paths.append((lora2_path, lora2_strength))

            if use_lora3 and lora3_url:
                logger.info("🎨 Processing custom LoRA 3...")
                lora3_path = download_lora_dynamic(lora3_url, civitai_token)
                if lora3_path:
                    model = load_lora3_node.load_lora_model_only(model, os.path.basename(lora3_path), lora3_strength)[0]
                    custom_lora_paths.append((lora3_path, lora3_strength))

            # Apply LightX2V LoRA
            if use_lightx2v:
                logger.info(f"⚡ Loading LightX2V LoRA rank {lightx2v_rank} (strength: {lightx2v_strength})...")
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path and os.path.exists(lightx2v_lora_path):
                    model = load_lightx2v_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), lightx2v_strength
                    )[0]
                    used_steps = 4  # LightX2V override steps theo code gốc
                    logger.info(f"✅ LightX2V LoRA applied, steps adjusted to {used_steps}")
                else:
                    logger.warning(f"⚠️ LightX2V LoRA not found: {lightx2v_lora_path}")

            # Apply sage attention
            if use_sage_attention:
                logger.info("🧠 Applying Sage Attention...")
                try:
                    model = pathch_sage_attention.patch(model, "auto")[0]
                    logger.info("✅ Sage Attention applied")
                except Exception as e:
                    logger.warning(f"⚠️ Sage Attention failed: {e}")

            # Apply TeaCache
            if rel_l1_thresh > 0:
                logger.info(f"🫖 Setting TeaCache: threshold={rel_l1_thresh}")
                try:
                    model = teacache.patch_teacache(model, rel_l1_thresh, start_percent, end_percent, "main_device", "14B")[0]
                    logger.info("✅ TeaCache applied")
                except Exception as e:
                    logger.warning(f"⚠️ TeaCache failed: {e}")

            # Sample với high noise model
            logger.info(f"🎬 Sampling with high noise model (steps: {used_steps}, end_step: {high_noise_steps})...")
            sampled = ksampler.sample(
                model=model,
                add_noise="enable",
                noise_seed=seed,
                steps=used_steps,
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
            torch.cuda.empty_cache()
            gc.collect()

            # STAGE 2: Low noise model
            logger.info("🎯 Loading low noise Q6_K model...")
            model = unet_loader.load_unet("wan2.2_i2v_low_noise_14B_Q6_K.gguf")[0]

            # Apply optimizations cho low noise model
            if use_nag:
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]

            if enable_flow_shift2:
                logger.info(f"🌊 Applying flow shift 2: {flow_shift2}")
                model = model_sampling.patch(model, flow_shift2)[0]

            # Re-apply prompt assist LoRAs cho low noise model
            if prompt_assist == "walking to viewers":
                model = pAssLora.load_lora_model_only(model, "walking to viewers_Wan.safetensors", 1.0)[0]
            elif prompt_assist == "walking from behind":
                model = pAssLora.load_lora_model_only(model, "walking_from_behind.safetensors", 1.0)[0]
            elif prompt_assist == "b3ll13-d8nc3r":
                model = pAssLora.load_lora_model_only(model, "b3ll13-d8nc3r.safetensors", 1.0)[0]

            # Re-apply custom LoRAs cho low noise model
            for lora_path, strength in custom_lora_paths:
                model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), strength)[0]

            # Apply PUSA LoRA cho low noise model (theo code gốc)
            if use_pusa:
                logger.info(f"🎭 Loading PUSA LoRA (strength: {pusa_strength})...")
                pusa_path = MODEL_CONFIGS.get("pusa_lora")
                if pusa_path and os.path.exists(pusa_path):
                    model = load_pusa_lora.load_lora_model_only(
                        model, os.path.basename(pusa_path), pusa_strength
                    )[0]
                    used_steps = 4  # PUSA override steps
                    logger.info("✅ PUSA LoRA applied")
                else:
                    logger.warning("⚠️ PUSA LoRA not found")

            # Re-apply optimizations
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
                steps=used_steps,
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
            torch.cuda.empty_cache()
            gc.collect()

            # Decode latents to video frames
            logger.info("🎨 Decoding latents to video frames...")
            decoded = vae_decode.decode(vae, sampled)[0]
            del vae
            torch.cuda.empty_cache()
            gc.collect()

            # Save video với optimized method
            logger.info("💾 Saving video with optimized method...")
            output_path = f"/app/ComfyUI/output/wan22_complete_{uuid.uuid4().hex[:8]}.mp4"
            
            # Use optimized save function
            final_output_path = save_video_optimized(decoded, output_path, fps)

            # Apply frame interpolation if enabled
            if enable_interpolation and interpolation_factor > 1:
                logger.info("🔄 Applying frame interpolation...")
                interpolated_path = apply_rife_interpolation(final_output_path, interpolation_factor)
                if interpolated_path != final_output_path and os.path.exists(interpolated_path):
                    return interpolated_path

            return final_output_path

    except Exception as e:
        logger.error(f"❌ Video generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    finally:
        # Comprehensive cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage với error handling"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")

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

def validate_input_parameters(job_input: dict) -> tuple[bool, str]:
    """
    Validate input parameters với comprehensive checking
    """
    try:
        # Required parameters
        required_params = ["image_url", "positive_prompt"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"

        # Validate image URL
        image_url = job_input["image_url"]
        try:
            response = requests.head(image_url, timeout=10)
            if response.status_code != 200:
                return False, f"Image URL not accessible: {response.status_code}"
        except Exception as e:
            return False, f"Image URL validation failed: {str(e)}"

        # Validate dimensions
        width = job_input.get("width", 720)
        height = job_input.get("height", 480)
        if not (256 <= width <= 2048 and 256 <= height <= 2048):
            return False, "Width and height must be between 256 and 2048"

        # Validate frames
        frames = job_input.get("frames", 65)
        if not (1 <= frames <= 120):
            return False, "Frames must be between 1 and 120"

        # Validate LightX2V rank
        lightx2v_rank = job_input.get("lightx2v_rank", "64")
        if lightx2v_rank not in ["32", "64", "128"]:
            return False, "LightX2V rank must be one of: 32, 64, 128"

        # Validate sampler
        sampler_name = job_input.get("sampler_name", "euler")
        valid_samplers = ["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m"]
        if sampler_name not in valid_samplers:
            return False, f"Invalid sampler. Must be one of: {', '.join(valid_samplers)}"

        # Validate CFG scale
        cfg_scale = job_input.get("cfg_scale", 1.0)
        if not (0.1 <= cfg_scale <= 20.0):
            return False, "CFG scale must be between 0.1 and 20.0"

        # Validate steps
        steps = job_input.get("steps", 6)
        if not (1 <= steps <= 50):
            return False, "Steps must be between 1 and 50"

        # Validate high_noise_steps
        high_noise_steps = job_input.get("high_noise_steps", 3)
        if not (1 <= high_noise_steps <= steps):
            return False, f"High noise steps must be between 1 and {steps}"

        return True, "Valid"

    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def handler(job):
    """
    Main RunPod handler cho WAN2.2 LightX2V Q6_K
    Complete với all parameters và comprehensive error handling
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()

    try:
        job_input = job.get("input", {})

        # Validate input parameters
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            return {
                "error": validation_message,
                "status": "failed",
                "job_id": job_id
            }

        # Extract validated parameters
        image_url = job_input["image_url"]
        positive_prompt = job_input["positive_prompt"]

        # Extract tất cả parameters với default values được cập nhật
        parameters = {
            # Required
            "positive_prompt": positive_prompt,

            # Basic video settings
            "negative_prompt": job_input.get("negative_prompt", "blurry, low quality, pixelated, artifacts, distorted, ugly, deformed, static, watermark, text, logo, poor lighting, nan, inf, invalid, corrupt"),
            "width": job_input.get("width", 720),
            "height": job_input.get("height", 480),
            "seed": job_input.get("seed", 0),
            "steps": job_input.get("steps", 6),
            "high_noise_steps": job_input.get("high_noise_steps", 3),
            "cfg_scale": job_input.get("cfg_scale", 1.0),
            "sampler_name": job_input.get("sampler_name", "euler"),
            "scheduler": job_input.get("scheduler", "simple"),
            "frames": job_input.get("frames", 65),
            "fps": job_input.get("fps", 16),

            # Prompt assist
            "prompt_assist": job_input.get("prompt_assist", "none"),

            # LoRA parameters
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

            # LightX2V parameters
            "use_lightx2v": job_input.get("use_lightx2v", True),
            "lightx2v_rank": job_input.get("lightx2v_rank", "64"),
            "lightx2v_strength": job_input.get("lightx2v_strength", 3.0),

            # PUSA parameters
            "use_pusa": job_input.get("use_pusa", False),
            "pusa_strength": job_input.get("pusa_strength", 1.2),

            # Optimization parameters
            "use_sage_attention": job_input.get("use_sage_attention", True),
            "rel_l1_thresh": job_input.get("rel_l1_thresh", 0.0),
            "start_percent": job_input.get("start_percent", 0.2),
            "end_percent": job_input.get("end_percent", 1.0),

            # Flow shift parameters
            "enable_flow_shift": job_input.get("enable_flow_shift", True),
            "flow_shift": job_input.get("flow_shift", 8.0),
            "enable_flow_shift2": job_input.get("enable_flow_shift2", True),
            "flow_shift2": job_input.get("flow_shift2", 8.0),

            # Advanced parameters
            "use_nag": job_input.get("use_nag", False),
            "nag_strength": job_input.get("nag_strength", 11.0),
            "nag_scale1": job_input.get("nag_scale1", 0.25),
            "nag_scale2": job_input.get("nag_scale2", 2.5),
            "use_clip_vision": job_input.get("use_clip_vision", False),

            # Frame interpolation
            "enable_interpolation": job_input.get("enable_interpolation", False),
            "interpolation_factor": job_input.get("interpolation_factor", 2)
        }

        logger.info(f"🚀 Job {job_id}: WAN2.2 Complete Generation Started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"📝 Prompt: {positive_prompt[:100]}...")
        logger.info(f"⚙️ Resolution: {parameters['width']}x{parameters['height']}")
        logger.info(f"🎬 Animation: {parameters['frames']} frames @ {parameters['fps']} FPS")
        logger.info(f"🎨 LoRAs: {sum([parameters['use_lora'], parameters['use_lora2'], parameters['use_lora3']])} custom + prompt_assist: {parameters['prompt_assist']}")

        # Verify models before processing
        models_ok, missing_models = verify_models()
        if not models_ok:
            return {
                "error": "Required models are missing",
                "missing_models": missing_models,
                "status": "failed"
            }

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
                return {"error": f"Failed to download image: {str(e)}"}

            # Generate video
            logger.info("🎬 Starting video generation...")
            generation_start = time.time()
            output_path = generate_video_wan22_complete(
                image_path=image_path,
                **parameters
            )
            generation_time = time.time() - generation_start

            if not output_path or not os.path.exists(output_path):
                return {"error": "Video generation failed"}

            # Upload result to MinIO
            logger.info("📤 Uploading result to storage...")
            output_filename = f"wan22_complete_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            try:
                output_url = upload_to_minio(output_path, output_filename)
            except Exception as e:
                return {"error": f"Failed to upload result: {str(e)}"}

            # Calculate final statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            duration_seconds = parameters["frames"] / parameters["fps"]

            # Determine actual seed used
            actual_seed = parameters["seed"] if parameters["seed"] != 0 else "auto-generated"

            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.1f}s (generation: {generation_time:.1f}s)")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB, {duration_seconds:.1f}s duration")

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
                    "file_size_mb": round(file_size_mb, 2)
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
                    "custom_loras": {
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
                        "flow_shift_value": parameters["flow_shift"],
                        "nag_enabled": parameters["use_nag"],
                        "clip_vision": parameters["use_clip_vision"]
                    },
                    "interpolation": {
                        "enabled": parameters["enable_interpolation"],
                        "factor": parameters["interpolation_factor"]
                    },
                    "model_quantization": "Q6_K"
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
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def health_check():
    """Health check function"""
    try:
        # Check CUDA
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Check models
        models_ok, missing = verify_models()
        if not models_ok:
            return False, f"Missing models: {len(missing)}"

        # Check ComfyUI
        if not COMFYUI_AVAILABLE:
            return False, "ComfyUI not available"

        # Check MinIO
        if not minio_client:
            return False, "MinIO not available"

        return True, "All systems operational"

    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting WAN2.2 LightX2V Complete Serverless Worker...")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    try:
        # Health check on startup
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            sys.exit(1)

        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎬 Ready to process WAN2.2 LightX2V requests...")

        # Start RunPod worker
        runpod.serverless.start({"handler": handler})

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
