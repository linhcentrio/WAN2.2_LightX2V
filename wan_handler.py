#!/usr/bin/env python3

"""
RunPod Serverless Handler cho WAN2.2 LightX2V Q6_K - OPTIMIZED VERSION
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add ComfyUI paths
sys.path.insert(0, '/app/ComfyUI')
sys.path.insert(0, '/app/Practical-RIFE')

# Import ComfyUI components
try:
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
    from comfy_extras.nodes_images import SaveAnimatedWEBP
    from comfy_extras.nodes_video import SaveWEBM
    from comfy_extras.nodes_wan import WanImageToVideo
    from comfy import model_management
    import folder_paths

    logger.info("✅ All ComfyUI modules imported successfully")
    COMFYUI_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ ComfyUI import error: {e}")
    COMFYUI_AVAILABLE = False

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client
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

# Model configurations CHÍNH XÁC theo file gốc
MODEL_CONFIGS = {
    # Q6_K DIT Models  
    "dit_model_high": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
    "dit_model_low": "/app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf",
    # Supporting models
    "text_encoder": "/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 
    "vae": "/app/ComfyUI/models/vae/wan_2.1_vae.safetensors",
    "clip_vision": "/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
    # LightX2V LoRAs theo rank - CHÍNH XÁC theo file gốc
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

# Global model cache
model_cache = {}

# Enable PyTorch optimizations 
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.flash_sdp_enabled = True
    torch.backends.cuda.mem_efficient_sdp_enabled = True
    logger.info("✅ PyTorch optimizations enabled")
except Exception as e:
    logger.warning(f"⚠️ PyTorch optimizations partially failed: {e}")

def verify_models() -> tuple[bool, list]:
    """Verify tất cả models cần thiết có tồn tại"""
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
        return False, missing_models
    else:
        logger.info(f"✅ All {len(existing_models)} models verified! Total: {total_size:.1f}MB") 
        return True, []

def download_lora_dynamic(lora_url: str, civitai_token: str = None) -> str:
    """Download LoRA từ HuggingFace hoặc CivitAI"""
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
    config_key = rank_mapping.get(lightx2v_rank, "lightx2v_rank_32")  # Default 32 theo file gốc
    return MODEL_CONFIGS.get(config_key)

def save_as_mp4_fixed(decoded, filename_prefix, fps, output_dir="/app/ComfyUI/output"):
    """
    ✅ FIXED: Save video với proper frame validation - Fix vấn đề 0MB
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"
    
    frames_processed = []
    
    try:
        for i, img in enumerate(decoded):
            # ✅ FIX: Validate frames trước khi convert
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy()
            else:
                img_np = np.array(img)
            
            # Remove NaN/Inf values - FIX CRITICAL BUG
            img_np = np.nan_to_num(img_np, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Clamp values to [0,1] range
            img_np = np.clip(img_np, 0.0, 1.0)
            
            # Convert to uint8 safely
            frame = (img_np * 255).astype(np.uint8)
            
            # Validate frame shape
            if len(frame.shape) == 3 and frame.shape[2] in [1, 3, 4]:
                frames_processed.append(frame)
            else:
                logger.warning(f"⚠️ Invalid frame {i} shape: {frame.shape}")
                
    except Exception as e:
        logger.error(f"❌ Frame processing error: {e}")
        raise RuntimeError(f"Failed to process frames: {e}")
    
    # Validate frames before writing
    if not frames_processed:
        raise ValueError("❌ No valid frames to save - video would be 0MB")
    
    try:
        # Use imageio with proper codec and bitrate
        with imageio.get_writer(
            output_path, 
            fps=fps, 
            codec='libx264',
            bitrate='8M',
            quality=8
        ) as writer:
            for frame in frames_processed:
                writer.append_data(frame)
                
        # Verify file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        if file_size == 0:
            raise ValueError("❌ Generated video is 0MB - encoding failed")
            
        logger.info(f"✅ Video saved successfully: {output_path} ({file_size:.1f}MB)")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Video encoding error: {e}")
        raise RuntimeError(f"Failed to encode video: {e}")

def generate_video_wan22_complete(image_path: str, **kwargs) -> str:
    """
    ✅ OPTIMIZED: Complete WAN2.2 video generation theo ĐÚNG logic file gốc
    """
    try:
        logger.info("🎬 Starting WAN2.2 Q6_K complete generation...")
        
        # Extract parameters với default values CHÍNH XÁC theo file gốc
        positive_prompt = kwargs.get('positive_prompt', '')
        negative_prompt = kwargs.get('negative_prompt', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
        
        # ✅ FIXED: Default resolution theo file gốc
        width = kwargs.get('width', 832)  # File gốc: 832x480
        height = kwargs.get('height', 480)
        
        seed = kwargs.get('seed', 0)
        # ✅ FIXED: Default steps theo file gốc 
        steps = kwargs.get('steps', 20)  # File gốc: 20 steps
        high_noise_steps = kwargs.get('high_noise_steps', 10)  # File gốc: 10
        cfg_scale = kwargs.get('cfg_scale', 1.0)
        sampler_name = kwargs.get('sampler_name', 'uni_pc')  # File gốc: uni_pc
        scheduler = kwargs.get('scheduler', 'simple')
        frames = kwargs.get('frames', 33)  # File gốc: 33 frames
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
        
        # ✅ FIXED: LightX2V configuration theo file gốc
        use_lightx2v = kwargs.get('use_lightx2v', True)  # File gốc: True
        lightx2v_rank = kwargs.get('lightx2v_rank', '32')  # File gốc: rank 32
        lightx2v_strength = kwargs.get('lightx2v_strength', 3.0)  # File gốc: 3.0
        
        # PUSA LoRA
        use_pusa = kwargs.get('use_pusa', False)
        pusa_strength = kwargs.get('pusa_strength', 1.2)
        
        # ✅ FIXED: Optimization parameters theo file gốc
        use_sage_attention = kwargs.get('use_sage_attention', True)
        rel_l1_thresh = kwargs.get('rel_l1_thresh', 0.275)  # File gốc: 0.275
        start_percent = kwargs.get('start_percent', 0.1)  # File gốc: 0.1  
        end_percent = kwargs.get('end_percent', 1.0)
        
        # Flow shift parameters
        enable_flow_shift = kwargs.get('enable_flow_shift', True)
        flow_shift = kwargs.get('flow_shift', 8.0)
        enable_flow_shift2 = kwargs.get('enable_flow_shift2', True)
        flow_shift2 = kwargs.get('flow_shift2', 8.0)
        
        # Advanced parameters
        use_nag = kwargs.get('use_nag', False)
        nag_strength = kwargs.get('nag_strength', 11.0)
        nag_scale1 = kwargs.get('nag_scale1', 0.25)
        nag_scale2 = kwargs.get('nag_scale2', 2.5)
        use_clip_vision = kwargs.get('use_clip_vision', False)  # File gốc: disabled
        
        # Frame interpolation
        enable_interpolation = kwargs.get('enable_interpolation', False)
        interpolation_factor = kwargs.get('interpolation_factor', 2)
        
        # Generate seed if auto
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
            
        logger.info(f"🎯 Generation Parameters (CHÍNH XÁC theo file gốc):")
        logger.info(f"   Resolution: {width}x{height}")
        logger.info(f"   Frames: {frames}, FPS: {fps}")
        logger.info(f"   Steps: {steps} (end_step1: {high_noise_steps})")
        logger.info(f"   CFG Scale: {cfg_scale}, Seed: {seed}")
        logger.info(f"   Sampler: {sampler_name} ({scheduler})")
        logger.info(f"   TeaCache: {rel_l1_thresh}")
        logger.info(f"   Prompt Assist: {prompt_assist}")
        logger.info(f"   LightX2V: {use_lightx2v} (rank: {lightx2v_rank}, strength: {lightx2v_strength})")
        
        # Verify ComfyUI availability
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI modules not available")
        
        # Auto-adjust height if 0 (maintain aspect ratio)
        if height == 0:
            from PIL import Image
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                height = int(width * original_height / original_width)
                height = (height // 8) * 8  # Ensure multiple of 8
            logger.info(f"🔄 Auto-adjusted height: {height} (aspect ratio preserved)")
        
        with torch.inference_mode(), autocast():
            # Initialize ComfyUI nodes
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
            
            # CLIP Vision processing (optional) - DISABLED theo file gốc
            clip_vision_output = None
            if use_clip_vision:
                logger.info("👁️ Processing with CLIP Vision...")
                clip_vision = clip_vision_loader.load_clip("clip_vision_h.safetensors")[0]
                clip_vision_output = clip_vision_encode.encode(clip_vision, loaded_image, "none")[0]
                del clip_vision
                torch.cuda.empty_cache()
                gc.collect()
            else:
                logger.info("👁️ CLIP Vision: disabled (theo file gốc)")
            
            # Load VAE
            logger.info("🎨 Loading VAE...")
            vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            
            # Encode image to video latent
            logger.info("🔄 Encoding image to video latent...")
            positive_out, negative_out, latent = wan_image_to_video.encode(
                positive, negative, vae, width, height, frames, 1, loaded_image, clip_vision_output
            )
            
            # STAGE 1: High noise model
            logger.info("🎯 STAGE 1: Loading high noise Q6_K model...")
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
                logger.info("🚶 Loading walking to camera LoRA...")
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
            
            # ✅ FIXED: Apply LightX2V LoRA theo file gốc
            if use_lightx2v:
                logger.info(f"⚡ Loading lightx2v LoRA rank {lightx2v_rank} (strength: {lightx2v_strength})...")
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
            logger.info(f"🎬 Generating video with high noise model (steps: {used_steps}, end_step: {high_noise_steps})...")
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
            logger.info("🎯 STAGE 2: Loading low noise Q6_K model...")
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
            logger.info(f"🎬 Generating video with low noise model (start_step: {high_noise_steps})...")
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
            logger.info("🎨 Decoding latents...")
            decoded = vae_decode.decode(vae, sampled)[0]
            
            del vae
            torch.cuda.empty_cache()
            gc.collect()
            
            # ✅ FIXED: Save video với proper validation
            logger.info("💾 Saving as MP4...")
            output_path = f"/app/ComfyUI/output/wan22_complete_{uuid.uuid4().hex[:8]}.mp4"
            output_path = save_as_mp4_fixed(decoded, f"wan22_complete_{uuid.uuid4().hex[:8]}", fps)
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ Video saved: {output_path} ({file_size_mb:.1f}MB)")
            
            # Apply frame interpolation if enabled
            if enable_interpolation and interpolation_factor > 1:
                logger.info("🔄 Applying frame interpolation...")
                interpolated_path = apply_rife_interpolation(output_path, interpolation_factor)
                if interpolated_path != output_path and os.path.exists(interpolated_path):
                    return interpolated_path
            
            return output_path
            
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
    """Upload file to MinIO storage"""
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

def handler(job):
    """
    ✅ OPTIMIZED: Main RunPod handler cho WAN2.2 LightX2V Q6_K
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Required inputs validation
        image_url = job_input.get("image_url")
        positive_prompt = job_input.get("positive_prompt", "")
        
        if not image_url:
            return {"error": "Missing required parameter: image_url"}
        if not positive_prompt:
            return {"error": "Missing required parameter: positive_prompt"}
        
        # Validate image URL
        try:
            response = requests.head(image_url, timeout=10)
            if response.status_code != 200:
                return {"error": f"Image URL not accessible: {response.status_code}"}
        except Exception as e:
            return {"error": f"Image URL validation failed: {str(e)}"}
        
        # ✅ FIXED: Extract parameters với default values CHÍNH XÁC theo file gốc
        parameters = {
            # Required
            "positive_prompt": positive_prompt,
            # Basic video settings với ĐÚNG default values
            "negative_prompt": job_input.get("negative_prompt", "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
            "width": job_input.get("width", 832),  # File gốc: 832
            "height": job_input.get("height", 480),  # File gốc: 480
            "seed": job_input.get("seed", 0),
            "steps": job_input.get("steps", 20),  # File gốc: 20
            "high_noise_steps": job_input.get("high_noise_steps", 10),  # File gốc: 10
            "cfg_scale": job_input.get("cfg_scale", 1.0),
            "sampler_name": job_input.get("sampler_name", "uni_pc"),  # File gốc: uni_pc
            "scheduler": job_input.get("scheduler", "simple"),
            "frames": job_input.get("frames", 33),  # File gốc: 33
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
            
            # ✅ FIXED: LightX2V parameters theo file gốc
            "use_lightx2v": job_input.get("use_lightx2v", True),  # File gốc: True
            "lightx2v_rank": job_input.get("lightx2v_rank", "32"),  # File gốc: rank 32
            "lightx2v_strength": job_input.get("lightx2v_strength", 3.0),  # File gốc: 3.0
            
            # PUSA parameters
            "use_pusa": job_input.get("use_pusa", False),
            "pusa_strength": job_input.get("pusa_strength", 1.2),
            
            # ✅ FIXED: Optimization parameters theo file gốc
            "use_sage_attention": job_input.get("use_sage_attention", True),
            "rel_l1_thresh": job_input.get("rel_l1_thresh", 0.275),  # File gốc: 0.275
            "start_percent": job_input.get("start_percent", 0.1),  # File gốc: 0.1
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
            "use_clip_vision": job_input.get("use_clip_vision", False),  # File gốc: disabled
            
            # Frame interpolation
            "enable_interpolation": job_input.get("enable_interpolation", False),
            "interpolation_factor": job_input.get("interpolation_factor", 2)
        }
        
        # Validate parameters
        if not (256 <= parameters["width"] <= 2048 and 256 <= parameters["height"] <= 2048):
            return {"error": "Width and height must be between 256 and 2048"}
        if not (1 <= parameters["frames"] <= 120):
            return {"error": "Frames must be between 1 and 120"}
        if parameters["lightx2v_rank"] not in ["32", "64", "128"]:
            return {"error": "LightX2V rank must be one of: 32, 64, 128"}
        
        logger.info(f"🚀 Job {job_id}: WAN2.2 Complete Generation Started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"📝 Prompt: {positive_prompt[:100]}...")
        logger.info(f"⚙️ Resolution: {parameters['width']}x{parameters['height']}")
        logger.info(f"🎬 Animation: {parameters['frames']} frames @ {parameters['fps']} FPS")
        logger.info(f"🎨 LightX2V: True (DEFAULT TRUE)")
        logger.info(f"🔧 Config: Theo đúng file gốc wan22_Lightx2v.ipynb")
        
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
