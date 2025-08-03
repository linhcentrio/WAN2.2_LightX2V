#!/usr/bin/env python3
"""
RunPod Handler WAN2.2 LightX2V Q6_K - Hoàn chỉnh theo code gốc
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import torch
import sys
import gc
import json
import random
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
from huggingface_hub import hf_hub_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add ComfyUI to path
sys.path.insert(0, '/app/ComfyUI')

# Import ComfyUI components
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

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Model configurations theo đúng code gốc
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
    
    # Built-in LoRAs
    "walking_to_viewers": "/app/ComfyUI/models/loras/walking to viewers_Wan.safetensors",
    "walking_from_behind": "/app/ComfyUI/models/loras/walking_from_behind.safetensors",
    "dancing": "/app/ComfyUI/models/loras/b3ll13-d8nc3r.safetensors",
    "pusa_lora": "/app/ComfyUI/models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors",
    "rotate_lora": "/app/ComfyUI/models/loras/rotate_20_epochs.safetensors"
}

# Enable PyTorch optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

def download_lora_dynamic(lora_url: str, civitai_token: str = None) -> str:
    """Download LoRA từ HuggingFace hoặc CivitAI theo code gốc"""
    try:
        lora_dir = "/app/ComfyUI/models/loras"
        
        if "huggingface.co" in lora_url:
            parts = lora_url.split("/")
            if len(parts) >= 6:
                username = parts[3]
                repo = parts[4]
                filename = parts[-1]
                
                local_path = os.path.join(lora_dir, filename)
                
                hf_hub_download(
                    repo_id=f"{username}/{repo}",
                    filename=filename,
                    local_dir=lora_dir,
                    force_download=True
                )
                
                logger.info(f"✅ Downloaded HF LoRA: {filename}")
                return local_path
                
        elif "civitai.com" in lora_url:
            if not civitai_token:
                raise ValueError("CivitAI token required for CivitAI downloads")
                
            try:
                model_id = lora_url.split("/models/")[1].split("?")[0]
            except IndexError:
                raise ValueError("Invalid CivitAI URL format")
                
            civitai_api_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
            if civitai_token:
                civitai_api_url += f"&token={civitai_token}"
                
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"civitai_model_{timestamp}.safetensors"
            local_path = os.path.join(lora_dir, filename)
            
            response = requests.get(civitai_api_url, timeout=300)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"✅ Downloaded CivitAI LoRA: {filename}")
            return local_path
        else:
            # Direct download
            filename = os.path.basename(urlparse(lora_url).path)
            if not filename.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                filename += '.safetensors'
                
            local_path = os.path.join(lora_dir, filename)
            
            response = requests.get(lora_url, timeout=300)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"✅ Downloaded LoRA: {filename}")
            return local_path
                
        return None
        
    except Exception as e:
        logger.error(f"❌ LoRA download failed: {e}")
        return None

def get_lightx2v_lora_path(lightx2v_rank: str) -> str:
    """Get LightX2V LoRA path theo rank"""
    rank_mapping = {
        "32": "lightx2v_rank_32",
        "64": "lightx2v_rank_64", 
        "128": "lightx2v_rank_128"
    }
    
    return MODEL_CONFIGS.get(rank_mapping.get(lightx2v_rank, "64"))

def generate_video_wan22_complete(image_path: str, **kwargs) -> str:
    """
    Complete WAN2.2 video generation theo đúng code gốc
    """
    try:
        logger.info("🎬 Starting WAN2.2 Q6_K complete generation...")
        
        # Extract all parameters từ code gốc
        positive_prompt = kwargs.get('positive_prompt', '')
        negative_prompt = kwargs.get('negative_prompt', '色调艳丽，过曝，静态，细节模糊不清')
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
        
        # Flow shift
        enable_flow_shift = kwargs.get('enable_flow_shift', True)
        flow_shift = kwargs.get('flow_shift', 8.0)
        enable_flow_shift2 = kwargs.get('enable_flow_shift2', True)  
        flow_shift2 = kwargs.get('flow_shift2', 8.0)
        
        # NAG parameters (disabled in original)
        use_nag = kwargs.get('use_nag', False)
        nag_strength = kwargs.get('nag_strength', 11.0)
        nag_scale1 = kwargs.get('nag_scale1', 0.25)
        nag_scale2 = kwargs.get('nag_scale2', 2.5)
        
        # CLIP Vision (disabled in original)
        use_clip_vision = kwargs.get('use_clip_vision', False)
        
        # Generate seed if needed
        if seed == 0:
            seed = random.randint(0, 2**32 - 1)
            
        logger.info(f"🎯 Parameters: {width}x{height}, {frames}f, {fps}fps, seed={seed}")
        
        # Modify positive prompt với prompt assist
        if prompt_assist != "none":
            positive_prompt = f"{positive_prompt} {prompt_assist}." if prompt_assist != "none" else positive_prompt
        
        with torch.inference_mode():
            # Initialize nodes
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
            
            # Load text encoder
            logger.info("📝 Loading Text Encoder...")
            clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            positive = clip_encode_positive.encode(clip, positive_prompt)[0]
            negative = clip_encode_negative.encode(clip, negative_prompt)[0]
            del clip
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load và process image
            logger.info("🖼️ Loading image...")
            loaded_image = load_image.load_image(image_path)[0]
            
            # Auto detect image dimensions nếu height = 0
            if height == 0:
                if loaded_image.ndim == 4:
                    _, height_int, width_int, _ = loaded_image.shape
                elif loaded_image.ndim == 3:
                    height_int, width_int, _ = loaded_image.shape
                else:
                    raise ValueError(f"Unsupported image shape: {loaded_image.shape}")
                height = int(width * height_int / width_int)
                
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
            positive_out, negative_out, latent = wan_image_to_video.encode(
                positive, negative, vae, width, height, frames, 1, loaded_image, clip_vision_output
            )
            
            # STAGE 1: High noise model
            logger.info("🎯 Loading high noise Q6_K model...")
            model = unet_loader.load_unet("wan2.2_i2v_high_noise_14B_Q6_K.gguf")[0]
            
            # Apply NAG nếu enabled (disabled in original)
            if use_nag:
                logger.info("🎯 Applying NAG...")
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]
            
            # Apply flow shift
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
            
            if use_lora and lora_url:
                logger.info("🎨 Downloading custom LoRA 1...")
                lora_path = download_lora_dynamic(lora_url, civitai_token)
                if lora_path:
                    model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), lora_strength)[0]
                    custom_lora_paths.append((lora_path, lora_strength))
                    
            if use_lora2 and lora2_url:
                logger.info("🎨 Downloading custom LoRA 2...")
                lora2_path = download_lora_dynamic(lora2_url, civitai_token)
                if lora2_path:
                    model = load_lora2_node.load_lora_model_only(model, os.path.basename(lora2_path), lora2_strength)[0]
                    custom_lora_paths.append((lora2_path, lora2_strength))
                    
            if use_lora3 and lora3_url:
                logger.info("🎨 Downloading custom LoRA 3...")
                lora3_path = download_lora_dynamic(lora3_url, civitai_token)
                if lora3_path:
                    model = load_lora3_node.load_lora_model_only(model, os.path.basename(lora3_path), lora3_strength)[0]
                    custom_lora_paths.append((lora3_path, lora3_strength))
            
            # Determine steps based on LoRA usage
            used_steps = steps
            
            # Apply LightX2V LoRA
            if use_lightx2v:
                logger.info(f"⚡ Loading LightX2V LoRA rank {lightx2v_rank} (strength: {lightx2v_strength})...")
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path:
                    model = load_lightx2v_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), lightx2v_strength
                    )[0]
                    used_steps = 4  # LightX2V override steps
            
            # Apply PUSA LoRA for stage 2 only (theo code gốc)
            
            # Apply sage attention
            if use_sage_attention:
                logger.info("🧠 Applying Sage Attention...")
                model = pathch_sage_attention.patch(model, "auto")[0]
            
            # Apply TeaCache
            if rel_l1_thresh > 0:
                logger.info(f"🫖 Setting TeaCache: {rel_l1_thresh}")
                model = teacache.patch_teacache(model, rel_l1_thresh, start_percent, end_percent, "main_device", "14B")[0]
            
            # Sample với high noise model
            logger.info("🎬 Generating với high noise model...")
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
            
            # Apply NAG for low noise (if enabled)
            if use_nag:
                model = wan_video_nag.patch(model, negative, nag_strength, nag_scale1, nag_scale2)[0]
            
            # Apply flow shift 2
            if enable_flow_shift2:
                model = model_sampling.patch(model, flow_shift2)[0]
            
            # Re-apply prompt assist LoRAs cho low noise model
            if prompt_assist == "walking to viewers":
                model = pAssLora.load_lora_model_only(model, "walking to viewers_Wan.safetensors", 1.0)[0]
            elif prompt_assist == "walking from behind":
                model = pAssLora.load_lora_model_only(model, "walking_from_behind.safetensors", 1.0)[0]
            elif prompt_assist == "b3ll13-d8nc3r":
                model = pAssLora.load_lora_model_only(model, "b3ll13-d8nc3r.safetensors", 1.0)[0]
            
            # Re-apply custom LoRAs
            for lora_path, strength in custom_lora_paths:
                if use_lora:
                    model = load_lora_node.load_lora_model_only(model, os.path.basename(lora_path), strength)[0]
            
            # Apply PUSA LoRA cho low noise model (theo code gốc)
            if use_pusa:
                logger.info(f"🎭 Loading PUSA LoRA (strength: {pusa_strength})...")
                # Trong code gốc, PUSA dùng lightx2v_lora path
                lightx2v_lora_path = get_lightx2v_lora_path(lightx2v_rank)
                if lightx2v_lora_path:
                    model = load_pusa_lora.load_lora_model_only(
                        model, os.path.basename(lightx2v_lora_path), pusa_strength
                    )[0]
                    used_steps = 4  # PUSA override steps
            
            # Re-apply optimizations
            if use_sage_attention:
                model = pathch_sage_attention.patch(model, "auto")[0]
                
            if rel_l1_thresh > 0:
                model = teacache.patch_teacache(model, rel_l1_thresh, start_percent, end_percent, "main_device", "14B")[0]
            
            # Sample với low noise model
            logger.info("🎬 Generating với low noise model...")
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
            
            # Decode latents
            logger.info("🎨 Decoding latents...")
            decoded = vae_decode.decode(vae, sampled)[0]
            del vae
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save video
            output_path = f"/app/ComfyUI/output/wan22_complete_{uuid.uuid4().hex[:8]}.mp4"
            
            import imageio
            frames = [(img.cpu().numpy() * 255).astype('uint8') for img in decoded]
            
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            logger.info(f"✅ Video generated: {output_path}")
            return output_path
            
    except Exception as e:
        logger.error(f"❌ Complete generation failed: {e}")
        return None
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

def handler(job):
    """Complete RunPod handler với tất cả parameters"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Required inputs
        image_url = job_input.get("image_url")
        positive_prompt = job_input.get("positive_prompt", "")
        
        if not image_url:
            return {"error": "Missing required parameter: image_url"}
            
        if not positive_prompt:
            return {"error": "Missing required parameter: positive_prompt"}
        
        # All parameters từ code gốc với default values
        parameters = {
            # Basic parameters
            "positive_prompt": positive_prompt,
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
            
            # Advanced parameters (mostly disabled in original)
            "use_nag": job_input.get("use_nag", False),
            "nag_strength": job_input.get("nag_strength", 11.0),
            "nag_scale1": job_input.get("nag_scale1", 0.25),
            "nag_scale2": job_input.get("nag_scale2", 2.5),
            
            "use_clip_vision": job_input.get("use_clip_vision", False)
        }
        
        logger.info(f"🚀 Job {job_id}: WAN2.2 Complete Generation")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"📝 Prompt: {positive_prompt[:100]}...")
        logger.info(f"⚙️ Parameters: {parameters['width']}x{parameters['height']}, {parameters['frames']}f, {parameters['fps']}fps")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download input image
            image_path = os.path.join(temp_dir, "input_image.jpg")
            logger.info("📥 Downloading input image...")
            
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Generate video
            logger.info("🎬 Starting complete video generation...")
            output_path = generate_video_wan22_complete(image_path=image_path, **parameters)
            
            if not output_path or not os.path.exists(output_path):
                return {"error": "Video generation failed"}
            
            # Upload result
            logger.info("📤 Uploading result...")
            output_filename = f"wan22_complete_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            
            minio_client.fput_object(MINIO_BUCKET, output_filename, output_path)
            output_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(output_filename)}"
            
            processing_time = time.time() - start_time
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "video_info": {
                    "width": parameters["width"],
                    "height": parameters["height"],
                    "frames": parameters["frames"],
                    "fps": parameters["fps"],
                    "duration_seconds": parameters["frames"] / parameters["fps"],
                    "file_size_mb": round(file_size, 2)
                },
                "generation_params": {
                    "positive_prompt": parameters["positive_prompt"],
                    "negative_prompt": parameters["negative_prompt"],
                    "steps": parameters["steps"],
                    "high_noise_steps": parameters["high_noise_steps"],
                    "cfg_scale": parameters["cfg_scale"],
                    "seed": parameters["seed"] if parameters["seed"] != 0 else "auto-generated",
                    "sampler_name": parameters["sampler_name"],
                    "scheduler": parameters["scheduler"],
                    "prompt_assist": parameters["prompt_assist"],
                    "lightx2v_used": parameters["use_lightx2v"],
                    "lightx2v_rank": parameters["lightx2v_rank"],
                    "lightx2v_strength": parameters["lightx2v_strength"],
                    "pusa_used": parameters["use_pusa"],
                    "custom_loras_used": {
                        "lora1": parameters["use_lora"],
                        "lora2": parameters["use_lora2"], 
                        "lora3": parameters["use_lora3"]
                    },
                    "optimizations": {
                        "sage_attention": parameters["use_sage_attention"],
                        "teacache": parameters["rel_l1_thresh"] > 0,
                        "flow_shift": parameters["enable_flow_shift"],
                        "nag": parameters["use_nag"],
                        "clip_vision": parameters["use_clip_vision"]
                    },
                    "model_quantization": "Q6_K"
                },
                "status": "completed"
            }
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("🚀 Starting WAN2.2 Complete Serverless Worker...")
    
    try:
        # Verify models
        missing_models = []
        for name, path in MODEL_CONFIGS.items():
            if not os.path.exists(path):
                missing_models.append(f"{name}: {path}")
        
        if missing_models:
            logger.error(f"❌ Missing models: {missing_models}")
            sys.exit(1)
            
        logger.info("✅ All models verified")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        sys.exit(1)
    
    logger.info("🎬 Ready to process complete WAN2.2 requests...")
    runpod.serverless.start({"handler": handler})
