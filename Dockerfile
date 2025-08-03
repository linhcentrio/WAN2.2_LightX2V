# Dockerfile với đầy đủ models theo code gốc
FROM spxiong/pytorch:2.6.0-py3.10.16-cuda12.6.0-ubuntu22.04

WORKDIR /app

# Environment
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    aria2 ffmpeg wget curl git \
    build-essential python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torchvision==0.21.0 torchsde==0.2.6 einops==0.8.0 \
    diffusers==0.32.2 transformers==4.48.0 accelerate==0.35.2 \
    xformers==0.0.29.post1 triton==3.2.0 sageattention \
    av spandrel albumentations insightface onnx opencv-python \
    segment_anything ultralytics onnxruntime-gpu \
    omegaconf==2.3.0 safetensors==0.4.5 tqdm==4.66.6 \
    psutil==6.0.0 kornia==0.8.0 soundfile==0.12.1 \
    runpod>=1.7.0 minio>=7.2.0 huggingface-hub==0.30.2

# Clone ComfyUI exactly như code gốc
RUN git clone --branch ComfyUI_v0.3.47 https://github.com/Isi-dev/ComfyUI /app/ComfyUI

# Clone custom nodes
RUN cd /app/ComfyUI/custom_nodes && \
    git clone https://github.com/Isi-dev/ComfyUI_GGUF.git && \
    git clone --branch kjnv1.1.3 https://github.com/Isi-dev/ComfyUI_KJNodes.git

# Install custom nodes requirements
RUN cd /app/ComfyUI/custom_nodes/ComfyUI_GGUF && pip install -r requirements.txt --no-cache-dir && \
    cd /app/ComfyUI/custom_nodes/ComfyUI_KJNodes && pip install -r requirements.txt --no-cache-dir

# Create directories
RUN mkdir -p /app/ComfyUI/models/{diffusion_models,text_encoders,vae,clip_vision,loras} && \
    mkdir -p /app/ComfyUI/{input,output,temp}

# Download ALL models theo đúng code gốc
RUN echo "=== Downloading Q6_K Models ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf" \
    -d /app/ComfyUI/models/diffusion_models && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf" \
    -d /app/ComfyUI/models/diffusion_models

# Supporting models
RUN echo "=== Supporting Models ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    -d /app/ComfyUI/models/text_encoders && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    -d /app/ComfyUI/models/vae && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    -d /app/ComfyUI/models/clip_vision

# LightX2V LoRAs ALL RANKS
RUN echo "=== LightX2V LoRAs ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors" \
    -d /app/ComfyUI/models/loras && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" \
    -d /app/ComfyUI/models/loras && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors" \
    -d /app/ComfyUI/models/loras

# Built-in LoRAs + PUSA
RUN echo "=== Built-in LoRAs ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking%20to%20viewers_Wan.safetensors" \
    -d /app/ComfyUI/models/loras -o "walking to viewers_Wan.safetensors" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking_from_behind.safetensors" \
    -d /app/ComfyUI/models/loras && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/b3ll13-d8nc3r.safetensors" \
    -d /app/ComfyUI/models/loras && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors" \
    -d /app/ComfyUI/models/loras && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Remade-AI/Rotate/resolve/main/rotate_20_epochs.safetensors" \
    -d /app/ComfyUI/models/loras

# Copy handler
COPY wan_handler.py /app/wan_handler.py

# Environment
ENV PYTHONPATH="/app:/app/ComfyUI"

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=240s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('🚀 Ready')" || exit 1

CMD ["python", "wan_handler.py"]
