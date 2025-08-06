#!/usr/bin/env python3

"""
RunPod Serverless Handler cho WAN2.2 LightX2V Q6_K - RIFE FIXED VERSION
Fixed RIFE interpolation using subprocess approach
Based on wan22_Lightx2v.ipynb workflow - All fixes and optimizations included
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
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
from huggingface_hub import hf_hub_download
import logging
import imageio
import numpy as np
from PIL import Image
import glob

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add ComfyUI paths
sys.path.insert(0, '/app/ComfyUI')
sys.path.insert(0, '/app/Practical-RIFE')

# Import ComfyUI components with comprehensive error handling
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

# Initialize MinIO client with error handling
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

# Model configurations with exact paths from notebook
MODEL_CONFIGS = {
    # Q6_K DIT Models (matching notebook defaults)
    "dit_model_high": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
    "dit_model_low": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf",
    
    # Supporting models (exact filenames from notebook)
    "text_encoder": "/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "/app/ComfyUI/models/vae/wan_2.1_vae.safetensors", 
    "clip_vision": "/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
    
    # LightX2V LoRAs by rank (matching notebook)
    "lightx2v_rank_32": "/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
    "lightx2v_rank_64": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
    "lightx2v_rank_128": "/app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
    
    # Built-in LoRAs with exact filenames from notebook
    "walking_to_viewers": "/app/ComfyUI/models/loras/walking to viewers_Wan.safetensors",
    "walking_from_behind": "/app/ComfyUI/models/loras/walking_from_behind.safetensors", 
    "dancing": "/app/ComfyUI/models/loras/b3ll13-d8nc3r.safetensors",
    "pusa_lora": "/app/ComfyUI/models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors",
    "rotate_lora": "/app/ComfyUI/models/loras/rotate_20_epochs.safetensors"
}

# Global model cache for memory optimization
model_cache = {}

# Enable PyTorch 2.6.0 optimizations (matching notebook)
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # Enable Flash Attention in PyTorch 2.6.0
    torch.backends.cuda.flash_sdp_enabled = True
    torch.backends.cuda.mem_efficient_sdp_enabled = True
    logger.info("✅ PyTorch 2.6.0 optimizations enabled")
except Exception as e:
    logger.warning(f"⚠️ PyTorch optimizations partially failed: {e}")

def verify_models() -> tuple[bool, list]:
    """Verify all required models exist - Returns: (success, missing_models)"""
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
    """Download LoRA from HuggingFace or CivitAI based on notebook logic"""
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

def calculate_aspect_preserving_dimensions(original_width, original_height, target_width, target_height):
    """ENHANCED: Calculate dimensions that preserve aspect ratio with padding"""
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height
    
    if original_ratio > target_ratio:
        # Image is wider -> padding top/bottom
        new_width = target_width
        new_height = int(target_width / original_ratio)
    else:
        # Image is taller -> padding left/right
        new_height = target_height
        new_width = int(target_height * original_ratio)
    
    # Ensure divisible by 8 for video encoding
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height

def calculate_crop_dimensions(original_width, original_height, target_width, target_height):
    """ENHANCED: Calculate smart crop to preserve ratio"""
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    
    if original_ratio > target_ratio:
        # Image is wider -> crop width
        crop_height = original_height
        crop_width = int(original_height * target_ratio)
        crop_x = (original_width - crop_width) // 2
        crop_y = 0
    else:
        # Image is taller -> crop height
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
    ENHANCED: Process image with different aspect ratio modes
    
    Args:
        aspect_mode:
            - "preserve": Keep aspect ratio, padding if needed
            - "crop": Smart crop to fit target ratio
            - "stretch": Stretch directly (like old workflow)
        auto_720p: Auto adjust to 720p standard
    """
    try:
        # Initialize nodes
        load_image = LoadImage()
        image_scaler = ImageScale()
        
        # Load original image
        loaded_image = load_image.load_image(image_path)[0]
        
        # Get original dimensions
        if loaded_image.ndim == 4:
            _, orig_height, orig_width, _ = loaded_image.shape
        else:
            orig_height, orig_width, _ = loaded_image.shape
            
        logger.info(f"🖼️ Original image: {orig_width}x{orig_height}")
        
        # Auto 720p detection
        if auto_720p:
            # Auto choose orientation based on original image
            if orig_width > orig_height:
                # Landscape -> 720p horizontal
                target_width, target_height = 1280, 720
            else:
                # Portrait -> 720p vertical
                target_width, target_height = 720, 1280
            logger.info(f"📱 Auto 720p mode: {target_width}x{target_height}")
        
        if aspect_mode == "preserve":
            # Keep aspect ratio with padding
            new_width, new_height = calculate_aspect_preserving_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            # Scale to calculated size
            scaled_image = image_scaler.upscale(
                loaded_image, "lanczos", new_width, new_height, "disabled"
            )[0]
            
            # Padding to reach target size if needed
            if new_width != target_width or new_height != target_height:
                # Create tensor with padding
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
            # Smart crop to preserve aspect ratio
            crop_x, crop_y, crop_width, crop_height = calculate_crop_dimensions(
                orig_width, orig_height, target_width, target_height
            )
            
            # Crop original image first
            if loaded_image.ndim == 3:
                cropped = loaded_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
            else:
                cropped = loaded_image[0, crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]
                cropped = cropped.unsqueeze(0)
            
            # Scale to target size
            final_image = image_scaler.upscale(
                cropped, "lanczos", target_width, target_height, "disabled"
            )[0]
            
            logger.info(f"✂️ Crop mode: {orig_width}x{orig_height} -> crop({crop_width}x{crop_height}) -> {target_width}x{target_height}")
            
        else:  # stretch
            # Direct stretch (old workflow)
            final_image = image_scaler.upscale(
                loaded_image, "lanczos", target_width, target_height, "disabled"
            )[0]
            
            logger.info(f"🔄 Stretch mode: {orig_width}x{orig_height} -> {target_width}x{target_height}")
        
        return final_image, target_width, target_height
        
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

def get_lightx2v_lora_path(lightx2v_rank: str) -> str:
    """Get LightX2V LoRA path by rank (exact from notebook)"""
    rank_mapping = {
        "32": "lightx2v_rank_32",
        "64": "lightx2v_rank_64", 
        "128": "lightx2v_rank_128"
    }
    
    config_key = rank_mapping.get(lightx2v_rank, "lightx2v_rank_32")
    return MODEL_CONFIGS.get(config_key)

def apply_rife_interpolation_subprocess(video_path: str, interpolation_factor: int = 2, target_fps: int = 30, crf_value: int = 17) -> str:
    """
    🔧 FIXED: Apply RIFE frame interpolation using subprocess approach
    Following exact logic from notebook with ffmpeg post-processing
    """
    try:
        logger.info(f"🔄 Applying RIFE interpolation (factor: {interpolation_factor}, target_fps: {target_fps}) via subprocess...")
        
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
            logger.warning("⚠️ Frame interpolation not available, returning original")
            return video_path

        # Setup environment variables to avoid ALSA errors (exact from notebook)
        env = os.environ.copy()
        env["XDG_RUNTIME_DIR"] = "/tmp"
        env["SDL_AUDIODRIVER"] = "dummy"
        env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        env["FFMPEG_LOGLEVEL"] = "quiet"

        # Build command to call RIFE script (exact from notebook)
        cmd = [
            "python3",
            "inference_video.py",
            f"--multi={interpolation_factor}",
            f"--fps={target_fps}",
            f"--video={video_path}",
            f"--scale=1"
        ]

        logger.info(f"🔧 Running RIFE command: {' '.join(cmd)}")
        logger.info(f"📍 Working directory: /app/Practical-RIFE")

        # Change to RIFE directory (critical step from notebook)
        original_cwd = os.getcwd()
        os.chdir("/app/Practical-RIFE")

        try:
            # Run RIFE with timeout protection
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout for long videos
                check=False
            )

            # Log output for debugging
            if result.stdout:
                logger.info(f"📝 RIFE stdout: {result.stdout[:1000]}...")
            if result.stderr:
                if "error" in result.stderr.lower() or "fail" in result.stderr.lower():
                    logger.warning(f"⚠️ RIFE stderr: {result.stderr[:1000]}...")
                else:
                    logger.info(f"📝 RIFE stderr: {result.stderr[:500]}...")

            if result.returncode != 0:
                logger.error(f"❌ RIFE process failed with return code: {result.returncode}")
                return video_path

        except subprocess.TimeoutExpired:
            logger.error("❌ RIFE process timed out")
            return video_path
        except Exception as e:
            logger.error(f"❌ RIFE subprocess failed: {e}")
            return video_path
        finally:
            # Restore original directory
            os.chdir(original_cwd)

        # Find generated interpolated file (following notebook logic)
        # RIFE usually creates file in /content/ComfyUI/output/ in notebook
        output_folder = "/app/ComfyUI/output/"
        
        # Find latest MP4 file created after running RIFE
        video_files = glob.glob(os.path.join(output_folder, "*.mp4"))
        
        if video_files:
            # Find latest file (created after running RIFE)
            latest_video = max(video_files, key=os.path.getctime)
            
            # Check if this file is new (created within last 5 minutes)
            if time.time() - os.path.getctime(latest_video) < 300:
                logger.info(f"🔍 Found RIFE interpolated file: {latest_video}")
                
                # Apply ffmpeg post-processing (exact from notebook)
                final_output_path = video_path.replace('.mp4', f'_rife_x{interpolation_factor}_converted.mp4')
                
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", latest_video,
                    "-vcodec", "libx264",
                    "-crf", str(crf_value),
                    "-preset", "fast",
                    final_output_path,
                    "-loglevel", "error",
                    "-y"  # Overwrite output file
                ]
                
                logger.info(f"🎬 Running ffmpeg post-processing...")
                try:
                    ffmpeg_result = subprocess.run(
                        ffmpeg_cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minutes timeout
                        check=True
                    )
                    
                    if os.path.exists(final_output_path):
                        final_size = os.path.getsize(final_output_path) / (1024 * 1024)
                        
                        # Validation check
                        if final_size < original_size * 0.3:  # Too small might indicate failure
                            logger.warning(f"⚠️ Final interpolated file suspiciously small: {final_size:.1f}MB vs {original_size:.1f}MB")
                            return video_path
                        
                        logger.info(f"✅ RIFE interpolation completed: {original_size:.1f}MB → {final_size:.1f}MB")
                        logger.info(f"📊 Output: {final_output_path}")
                        return final_output_path
                    else:
                        logger.error("❌ FFmpeg failed to create output file")
                        return video_path
                        
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ FFmpeg failed: {e}")
                    logger.error(f"FFmpeg stderr: {e.stderr}")
                    return video_path
                except subprocess.TimeoutExpired:
                    logger.error("❌ FFmpeg timed out")
                    return video_path
            else:
                logger.warning("⚠️ No recent interpolated file found")
                return video_path
        else:
            logger.warning("⚠️ No video files found in output directory")
            return video_path

    except Exception as e:
        logger.error(f"❌ RIFE interpolation error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return video_path

def clear_memory():
    """Enhanced memory cleanup matching notebook"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Additional CUDA memory cleanup
        try:
            torch.cuda.synchronize()
        except:
            pass

def save_video_optimized(frames_tensor, output_path, fps=16):
    """
    FINAL video saving function - no minimum size validation
    Based on notebook workflow with simplified validation
    """
    try:
        logger.info(f"🎬 Saving video with FINAL OPTIMIZED method...")
        logger.info(f"📍 Output path: {output_path}")
        
        # Validate input
        if frames_tensor is None:
            raise ValueError("Frames tensor is None")
        
        # Convert tensor to numpy - exactly like notebook
        if torch.is_tensor(frames_tensor):
            frames_np = frames_tensor.detach().cpu().float().numpy()
        else:
            frames_np = np.array(frames_tensor, dtype=np.float32)
            
        logger.info(f"📊 Frames tensor shape: {frames_np.shape}, dtype: {frames_np.dtype}")
        
        # Handle batch dimension if present
        if frames_np.ndim == 5 and frames_np.shape[0] == 1:
            frames_np = frames_np[0]  # Remove batch dimension
        
        # Convert to uint8 (0-255 range) - same as notebook
        logger.info("🔄 Converting to uint8 format...")
        frames_np = (frames_np * 255.0).astype(np.uint8)
        
        logger.info(f"📊 Final video frames stats:")
        logger.info(f"  Shape: {frames_np.shape}")
        logger.info(f"  Type: {frames_np.dtype}")
        logger.info(f"  Value range: [{frames_np.min()}, {frames_np.max()}]")
        
        # Validate frame dimensions
        if len(frames_np.shape) != 4:  # Should be [frames, height, width, channels]
            raise ValueError(f"Invalid frame shape: {frames_np.shape} (expected 4D)")
            
        num_frames, height, width, channels = frames_np.shape
        
        if num_frames == 0:
            raise ValueError("No frames to save")
            
        if channels not in [1, 3, 4]:
            raise ValueError(f"Invalid number of channels: {channels}")
        
        # Convert to RGB if needed (same as notebook workflow)
        if channels == 1:
            logger.info("🔄 Converting grayscale to RGB...")
            frames_np = np.repeat(frames_np, 3, axis=-1)
        elif channels == 4:
            logger.info("🔄 Removing alpha channel...")
            frames_np = frames_np[:, :, :, :3]  # Remove alpha channel
            
        # Update channels after conversion
        channels = frames_np.shape[-1]
        
        logger.info(f"📊 Final format: {num_frames} frames, {height}x{width}, {channels} channels")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save video using imageio (EXACT same as notebook)
        logger.info("💾 Writing video file...")
        frames_list = [frame for frame in frames_np]
        
        try:
            with imageio.get_writer(output_path, fps=fps) as writer:
                for i, frame in enumerate(frames_list):
                    writer.append_data(frame)
                    # Log progress every 10 frames
                    if (i + 1) % 10 == 0:
                        logger.info(f"📹 Written frame {i + 1}/{num_frames}")
                        
        except Exception as write_error:
            logger.error(f"❌ Primary video write failed: {write_error}")
            logger.info("🔄 Trying alternative video encoding...")
            
            # Alternative encoding method (fallback)
            with imageio.get_writer(
                output_path, 
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p'
            ) as writer:
                for i, frame in enumerate(frames_list):
                    writer.append_data(frame)
                    if (i + 1) % 10 == 0:
                        logger.info(f"📹 (Alt) Written frame {i + 1}/{num_frames}")
        
        # Verify file was created (NO SIZE VALIDATION)
        if not os.path.exists(output_path):
            raise FileNotFoundError("Video file was not created")
            
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        # SIMPLIFIED - just log stats, no validation
        logger.info(f"📊 Video file stats:")
        logger.info(f"  Size: {file_size_mb:.2f}MB")
        logger.info(f"  Frames: {num_frames}, Resolution: {width}x{height}")
        
        logger.info(f"✅ Video saved successfully: {output_path}")
        logger.info(f"📊 Final size: {file_size_mb:.2f}MB for {num_frames} frames @ {fps}fps")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ FINAL video saving failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

def generate_video_wan22_complete(image_path: str, **kwargs) -> str:
    """
    ENHANCED Complete WAN2.2 video generation with aspect ratio control
    FINAL VERSION - All optimizations and fixes included with enhanced image processing
    """
    try:
        logger.info("🎬 Starting WAN2.2 Q6_K ENHANCED generation (notebook workflow)...")
        
        # Extract parameters with EXACT default values from notebook
        positive_prompt = kwargs.get('positive_prompt', '')
        negative_prompt = kwargs.get('negative_prompt', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
        
        # ENHANCED: Aspect ratio control parameters
        aspect_mode = kwargs.get('aspect_mode', 'preserve')  # preserve, crop, stretch
        auto_720p = kwargs.get('auto_720p', False)
        
        # Video settings (matching notebook defaults)
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
        
        # LightX2V configuration (matching notebook)
        use_lightx2v = kwargs.get('use_lightx2v', True)
        lightx2v_rank = kwargs.get('lightx2v_rank', '32')  # Default to 32 like notebook
        lightx2v_strength = kwargs.get('lightx2v_strength', 3.0)
        
        # PUSA LoRA (matching notebook usage)
        use_pusa = kwargs.get('use_pusa', True)  # Default True for stage 2
        pusa_strength = kwargs.get('pusa_strength', 1.5)
        
        # Optimization parameters (conservative defaults)
        use_sage_attention = kwargs.get('use_sage_attention', True)
        rel_l1_thresh = kwargs.get('rel_l1_thresh', 0.0)  # Default 0 for stability
        start_percent = kwargs.get('start_percent', 0.2)
        end_percent = kwargs.get('end_percent', 1.0)
        
        # Flow shift parameters (matching notebook)
        enable_flow_shift = kwargs.get('enable_flow_shift', True)
        flow_shift = kwargs.get('flow_shift', 8.0)
        enable_flow_shift2 = kwargs.get('enable_flow_shift2', True)
        flow_shift2 = kwargs.get('flow_shift2', 8.0)
        
        # Advanced parameters (disabled by default for stability)
        use_nag = kwargs.get('use_nag', False)
        nag_strength = kwargs.get('nag_strength', 11.0)
        nag_scale1 = kwargs.get('nag_scale1', 0.25)
        nag_scale2 = kwargs.get('nag_scale2', 2.5)
        use_clip_vision = kwargs.get('use_clip_vision', False)
        
        # Frame interpolation
        enable_interpolation = kwargs.get('enable_interpolation', False)
        interpolation_factor = kwargs.get('interpolation_factor', 2)
        interpolation_fps = kwargs.get('interpolation_fps', 30)  # Target FPS after interpolation
        crf_value = kwargs.get('crf_value', 17)  # CRF value for ffmpeg
        
        # Generate seed if auto
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
            
        logger.info(f"🎯 Generation Parameters:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  Aspect Mode: {aspect_mode}, Auto 720p: {auto_720p}")
        logger.info(f"  Frames: {frames}, FPS: {fps}")
        logger.info(f"  Steps: {steps} (high noise: {high_noise_steps})")
        logger.info(f"  CFG Scale: {cfg_scale}, Seed: {seed}")
        logger.info(f"  Prompt Assist: {prompt_assist}")
        logger.info(f"  LightX2V: {use_lightx2v} (rank: {lightx2v_rank}, strength: {lightx2v_strength})")
        logger.info(f"  PUSA: {use_pusa} (strength: {pusa_strength})")
        logger.info(f"  Frame Interpolation: {enable_interpolation} (factor: {interpolation_factor})")
        
        # Verify ComfyUI availability
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI modules not available")
            
        with torch.inference_mode():
            # Initialize all ComfyUI nodes (EXACT like notebook)
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
            
            logger.info("✅ ComfyUI nodes initialized")
            
            # Load text encoder (EXACT like notebook)
            logger.info("📝 Loading Text_Encoder...")
            clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            
            # Modify prompt with prompt assist
            final_positive_prompt = positive_prompt
            if prompt_assist != "none":
                final_positive_prompt = f"{positive_prompt} {prompt_assist}."
                
            positive = clip_encode_positive.encode(clip, final_positive_prompt)[0]
            negative = clip_encode_negative.encode(clip, negative_prompt)[0]
            
            del clip
            clear_memory()
            
            # ENHANCED: Load and process image with aspect control
            logger.info("🖼️ Loading and processing image with ENHANCED aspect control...")
            loaded_image, final_width, final_height = process_image_with_aspect_control(
                image_path=image_path,
                target_width=width,
                target_height=height,
                aspect_mode=aspect_mode,
                auto_720p=auto_720p
            )
            
            # Update dimensions for video generation
            width, height = final_width, final_height
            
            # CLIP Vision processing (optional, disabled by default like notebook)
            clip_vision_output = None
            if use_clip_vision:
                logger.info("👁️ Processing with CLIP Vision...")
                clip_vision = clip_vision_loader.load_clip("clip_vision_h.safetensors")[0]
                clip_vision_output = clip_vision_encode.encode(clip_vision, loaded_image, "none")[0]
                del clip_vision
                clear_memory()
            
            # Load VAE (EXACT like notebook)
            logger.info("🎨 Loading VAE...")
            vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            
            # Encode image to video latent (EXACT like notebook)
            logger.info("🔄 Encoding image to video latent...")
            positive_out, negative_out, latent = wan_image_to_video.encode(
                positive, negative, vae, width, height, frames, 1, loaded_image, clip_vision_output
            )
            
            # STAGE 1: High noise model (EXACT workflow from notebook)
            logger.info("🎯 Loading high noise Model...")
            model = unet_loader.load_unet("wan2.2_i2v_high_noise_14B_Q6_K.gguf")[0]
            
            # Apply NAG if enabled (disabled by default)
            if use_nag:
                logger.info(f"🎯 Applying NAG (strength: {nag_strength})...")
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]
            
            # Apply flow shift for high noise (matching notebook)
            if enable_flow_shift:
                logger.info(f"🌊 Applying flow shift: {flow_shift}")
                model = model_sampling.patch(model, flow_shift)[0]
                
            # Apply prompt assist LoRAs (EXACT like notebook)
            if prompt_assist != "none":
                if prompt_assist == "walking to viewers":
                    logger.info("🚶 Loading walking to viewers LoRA...")
                    model = pAssLora.load_lora_model_only(model, "walking to viewers_Wan.safetensors", 1.0)[0]
                elif prompt_assist == "walking from behind":
                    logger.info("🚶 Loading walking from behind LoRA...")
                    model = pAssLora.load_lora_model_only(model, "walking_from_behind.safetensors", 1.0)[0]
                elif prompt_assist == "b3ll13-d8nc3r":
                    logger.info("💃 Loading dancing LoRA...")
                    model = pAssLora.load_lora_model_only(model, "b3ll13-d8nc3r.safetensors", 1.0)[0]
            
            # Download and apply custom LoRAs
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
            
            # Apply LightX2V LoRA (EXACT logic from notebook)
            if use_lightx2v:
                logger.info(f"⚡ Loading LightX2V LoRA rank {lightx2v_rank} (strength: {lightx2v_strength})...")
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path and os.path.exists(lightx2v_lora_path):
                    model = load_lightx2v_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), lightx2v_strength
                    )[0]
                    used_steps = steps  # Keep original steps
                    logger.info(f"✅ LightX2V LoRA applied")
                else:
                    logger.warning(f"⚠️ LightX2V LoRA not found: {lightx2v_lora_path}")
            
            # Apply sage attention (default enabled)
            if use_sage_attention:
                logger.info("🧠 Applying Sage Attention...")
                try:
                    model = pathch_sage_attention.patch(model, "auto")[0]
                    logger.info("✅ Sage Attention applied")
                except Exception as e:
                    logger.warning(f"⚠️ Sage Attention failed: {e}")
            
            # Apply TeaCache (only if threshold > 0)
            if rel_l1_thresh > 0:
                logger.info(f"🫖 Setting TeaCache: threshold={rel_l1_thresh}")
                try:
                    model = teacache.patch_teacache(model, rel_l1_thresh, start_percent, end_percent, "main_device", "14B")[0]
                    logger.info("✅ TeaCache applied")
                except Exception as e:
                    logger.warning(f"⚠️ TeaCache failed: {e}")
            
            # Sample with high noise model (EXACT like notebook)
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
            clear_memory()
            
            # STAGE 2: Low noise model (EXACT workflow from notebook)
            logger.info("🎯 Loading low noise Model...")
            model = unet_loader.load_unet("wan2.2_i2v_low_noise_14B_Q6_K.gguf")[0]
            
            # Apply optimizations for low noise model
            if use_nag:
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]
                
            if enable_flow_shift2:
                logger.info(f"🌊 Applying flow shift 2: {flow_shift2}")
                model = model_sampling.patch(model, flow_shift2)[0]
            
            # Re-apply prompt assist LoRAs for low noise model
            if prompt_assist != "none":
                if prompt_assist == "walking to viewers":
                    model = pAssLora.load_lora_model_only(model, "walking to viewers_Wan.safetensors", 1.0)[0]
                elif prompt_assist == "walking from behind":
                    model = pAssLora.load_lora_model_only(model, "walking_from_behind.safetensors", 1.0)[0]
                elif prompt_assist == "b3ll13-d8nc3r":
                    model = pAssLora.load_lora_model_only(model, "b3ll13-d8nc3r.safetensors", 1.0)[0]
            
            # Re-apply custom LoRAs for low noise model
            for lora_path, strength in custom_lora_paths:
                model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), strength)[0]
            
            # Apply PUSA LoRA for low noise model (EXACT like notebook logic)
            if use_pusa:
                logger.info(f"🎭 Loading PUSA LoRA (strength: {pusa_strength})...")
                # Use lightx2v_lora for PUSA like in notebook
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path and os.path.exists(lightx2v_lora_path):
                    model = load_pusa_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), pusa_strength
                    )[0]
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
            
            # Sample with low noise model (EXACT like notebook)
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
            clear_memory()
            
            # Decode latents to video frames (EXACT like notebook)
            logger.info("🎨 Decoding latents...")
            decoded = vae_decode.decode(vae, sampled)[0]
            
            del vae
            clear_memory()
            
            # Save video with FINAL optimized method
            logger.info("💾 Saving video with FINAL OPTIMIZED method...")
            output_path = f"/app/ComfyUI/output/wan22_enhanced_{uuid.uuid4().hex[:8]}.mp4"
            
            # Use the FINAL save function
            final_output_path = save_video_optimized(decoded, output_path, fps)
            
            # 🔧 FIXED: FRAME INTERPOLATION - Apply if enabled with subprocess approach
            if enable_interpolation and interpolation_factor > 1:
                logger.info(f"🔄 Applying frame interpolation (factor: {interpolation_factor})...")
                
                # Log current video stats before interpolation
                pre_interp_size = os.path.getsize(final_output_path) / (1024 * 1024)
                logger.info(f"📊 Pre-interpolation: {pre_interp_size:.1f}MB")

                # Apply interpolation with updated function
                interpolated_path = apply_rife_interpolation_subprocess(
                    final_output_path, 
                    interpolation_factor, 
                    interpolation_fps, 
                    crf_value
                )

                # Check if interpolation was successful
                if interpolated_path != final_output_path and os.path.exists(interpolated_path):
                    post_interp_size = os.path.getsize(interpolated_path) / (1024 * 1024)
                    logger.info(f"📊 Post-interpolation: {post_interp_size:.1f}MB")
                    logger.info(f"✅ Frame interpolation successful, using interpolated video")
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

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage with error handling"""
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
    """ENHANCED Validate input parameters with comprehensive checking"""
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
        
        # Validate dimensions (conservative limits)
        width = job_input.get("width", 720)
        height = job_input.get("height", 1280)
        if not (256 <= width <= 1024 and 256 <= height <= 1536):
            return False, "Width must be 256-1024, height must be 256-1536"
        
        # Validate frames (conservative limit)
        frames = job_input.get("frames", 65)
        if not (1 <= frames <= 100):
            return False, "Frames must be between 1 and 100"
        
        # ENHANCED: Validate aspect_mode
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
        
        # Validate CFG scale
        cfg_scale = job_input.get("cfg_scale", 1.0)
        if not (0.1 <= cfg_scale <= 10.0):
            return False, "CFG scale must be between 0.1 and 10.0"
        
        # Validate steps
        steps = job_input.get("steps", 6)
        if not (1 <= steps <= 30):
            return False, "Steps must be between 1 and 30"
        
        # Validate high_noise_steps
        high_noise_steps = job_input.get("high_noise_steps", 3)
        if not (1 <= high_noise_steps < steps):
            return False, f"High noise steps must be between 1 and {steps-1}"
        
        # Validate interpolation factor
        interpolation_factor = job_input.get("interpolation_factor", 2)
        if not (2 <= interpolation_factor <= 8):
            return False, "Interpolation factor must be between 2 and 8"
        
        # Validate interpolation FPS
        interpolation_fps = job_input.get("interpolation_fps", 30)
        if not (10 <= interpolation_fps <= 60):
            return False, "Interpolation FPS must be between 10 and 60"

        # Validate CRF value
        crf_value = job_input.get("crf_value", 17)
        if not (0 <= crf_value <= 51):
            return False, "CRF value must be between 0 and 51"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def handler(job):
    """
    ENHANCED Main RunPod handler for WAN2.2 LightX2V Q6_K
    RIFE FIXED VERSION with subprocess approach - All fixes and optimizations included
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
        
        # Extract validated parameters with notebook defaults
        image_url = job_input["image_url"]
        positive_prompt = job_input["positive_prompt"]
        
        # Extract all parameters with EXACT default values from notebook
        parameters = {
            # Required
            "positive_prompt": positive_prompt,
            
            # Basic video settings (matching notebook defaults)
            "negative_prompt": job_input.get("negative_prompt", "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
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
            
            # ENHANCED: Aspect ratio control
            "aspect_mode": job_input.get("aspect_mode", "preserve"),  # preserve, crop, stretch
            "auto_720p": job_input.get("auto_720p", False),
            
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
            
            # LightX2V parameters (matching notebook)
            "use_lightx2v": job_input.get("use_lightx2v", True),
            "lightx2v_rank": job_input.get("lightx2v_rank", "32"),  # Default to 32
            "lightx2v_strength": job_input.get("lightx2v_strength", 3.0),
            
            # PUSA parameters (matching notebook)
            "use_pusa": job_input.get("use_pusa", True),  # Default True for stage 2
            "pusa_strength": job_input.get("pusa_strength", 1.5),
            
            # Optimization parameters (conservative)
            "use_sage_attention": job_input.get("use_sage_attention", True),
            "rel_l1_thresh": job_input.get("rel_l1_thresh", 0.0),  # Default 0
            "start_percent": job_input.get("start_percent", 0.2),
            "end_percent": job_input.get("end_percent", 1.0),
            
            # Flow shift parameters (matching notebook)
            "enable_flow_shift": job_input.get("enable_flow_shift", True),
            "flow_shift": job_input.get("flow_shift", 8.0),
            "enable_flow_shift2": job_input.get("enable_flow_shift2", True),
            "flow_shift2": job_input.get("flow_shift2", 8.0),
            
            # Advanced parameters (disabled by default)
            "use_nag": job_input.get("use_nag", False),
            "nag_strength": job_input.get("nag_strength", 11.0),
            "nag_scale1": job_input.get("nag_scale1", 0.25),
            "nag_scale2": job_input.get("nag_scale2", 2.5),
            "use_clip_vision": job_input.get("use_clip_vision", False),
            
            # Frame interpolation (enhanced parameters)
            "enable_interpolation": job_input.get("enable_interpolation", False),
            "interpolation_factor": job_input.get("interpolation_factor", 2),
            "interpolation_fps": job_input.get("interpolation_fps", 30),  # Target FPS after interpolation
            "crf_value": job_input.get("crf_value", 17),  # CRF value for ffmpeg
        }
        
        logger.info(f"🚀 Job {job_id}: WAN2.2 RIFE FIXED Generation Started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"📝 Prompt: {positive_prompt[:100]}...")
        logger.info(f"⚙️ Resolution: {parameters['width']}x{parameters['height']}")
        logger.info(f"🎨 Aspect Mode: {parameters['aspect_mode']}, Auto 720p: {parameters['auto_720p']}")
        logger.info(f"🎬 Animation: {parameters['frames']} frames @ {parameters['fps']} FPS")
        logger.info(f"🎨 LightX2V: rank {parameters['lightx2v_rank']}, strength {parameters['lightx2v_strength']}")
        logger.info(f"🔄 Interpolation: {parameters['enable_interpolation']} (factor: {parameters['interpolation_factor']})")
        
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
            
            # Generate video with ENHANCED notebook workflow
            logger.info("🎬 Starting RIFE FIXED video generation (notebook workflow)...")
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
            output_filename = f"wan22_enhanced_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            
            try:
                output_url = upload_to_minio(output_path, output_filename)
            except Exception as e:
                return {"error": f"Failed to upload result: {str(e)}"}
            
            # Calculate final statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            duration_seconds = parameters["frames"] / parameters["fps"]
            
            # If interpolation was used, adjust duration
            if parameters["enable_interpolation"] and output_path.endswith(f"_rife_x{parameters['interpolation_factor']}_converted.mp4"):
                duration_seconds = (parameters["frames"] * parameters["interpolation_factor"]) / (parameters["interpolation_fps"])
            
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
                    "file_size_mb": round(file_size_mb, 2),
                    "interpolated": parameters["enable_interpolation"],
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
                        "factor": parameters["interpolation_factor"],
                        "target_fps": parameters["interpolation_fps"],
                        "crf_value": parameters["crf_value"]
                    },
                    "aspect_control": {
                        "mode": parameters["aspect_mode"],
                        "auto_720p": parameters["auto_720p"]
                    },
                    "model_quantization": "Q6_K",
                    "workflow_version": "RIFE_FIXED_COMPLETE"
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
    logger.info("🚀 Starting WAN2.2 LightX2V RIFE FIXED Serverless Worker...")
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
        logger.info("🎬 Ready to process WAN2.2 LightX2V requests (RIFE FIXED VERSION)...")
        logger.info("🔧 All fixes, optimizations, aspect ratio control and RIFE fix included - Production ready!")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
