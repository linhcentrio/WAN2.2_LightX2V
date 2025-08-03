# WAN2.2 LightX2V Docker Image - Fixed Version
FROM spxiong/pytorch:2.6.0-py3.10.16-cuda12.6.0-ubuntu22.04

WORKDIR /app

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    aria2 ffmpeg wget curl git \
    build-essential python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# BƯỚC 1: Cài đặt PyTorch ecosystem với version fix
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -U \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    xformers \
    --index-url https://download.pytorch.org/whl/cu126

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# BƯỚC 2: Cài đặt Triton và SageAttention với version fix
RUN pip install --no-cache-dir triton==3.2.0 && \
    pip install --no-cache-dir sageattention || \
    echo "⚠️ SageAttention installation failed, continuing without it"

# BƯỚC 3: Copy requirements và cài đặt dependencies cơ bản
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# BƯỚC 4: Clone ComfyUI và custom nodes
RUN git clone --branch ComfyUI_v0.3.47 https://github.com/Isi-dev/ComfyUI /app/ComfyUI && \
    cd /app/ComfyUI/custom_nodes && \
    git clone https://github.com/Isi-dev/ComfyUI_GGUF.git && \
    git clone --branch kjnv1.1.3 https://github.com/Isi-dev/ComfyUI_KJNodes.git

# BƯỚC 5: Install custom nodes requirements
RUN cd /app/ComfyUI/custom_nodes/ComfyUI_GGUF && \
    pip install --no-cache-dir -r requirements.txt || echo "⚠️ GGUF requirements failed" && \
    cd /app/ComfyUI/custom_nodes/ComfyUI_KJNodes && \
    pip install --no-cache-dir -r requirements.txt || echo "⚠️ KJNodes requirements failed"

# BƯỚC 6: Setup Practical-RIFE cho frame interpolation (thiếu trong dockerfile cũ)
RUN git clone https://github.com/Isi-dev/Practical-RIFE /app/Practical-RIFE && \
    cd /app/Practical-RIFE && \
    pip install --no-cache-dir git+https://github.com/rk-exxec/scikit-video.git@numpy_deprecation && \
    mkdir -p /app/Practical-RIFE/train_log

# Download Practical-RIFE models
RUN cd /app/Practical-RIFE && \
    wget -q https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/IFNet_HDv3.py \
    -O /app/Practical-RIFE/train_log/IFNet_HDv3.py && \
    wget -q https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/RIFE_HDv3.py \
    -O /app/Practical-RIFE/train_log/RIFE_HDv3.py && \
    wget -q https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/refine.py \
    -O /app/Practical-RIFE/train_log/refine.py && \
    wget -q https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/flownet.pkl \
    -O /app/Practical-RIFE/train_log/flownet.pkl

# BƯỚC 7: Create directories
RUN mkdir -p /app/ComfyUI/models/{diffusion_models,text_encoders,vae,clip_vision,loras} && \
    mkdir -p /app/ComfyUI/{input,output,temp}

# BƯỚC 8: Download Q6_K Models
RUN echo "=== Downloading Q6_K Models ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf" \
    -d /app/ComfyUI/models/diffusion_models && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf" \
    -d /app/ComfyUI/models/diffusion_models

# Download supporting models
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

# Download LightX2V LoRAs
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

# Download built-in LoRAs
RUN echo "=== Built-in LoRAs ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking%20to%20viewers_Wan.safetensors" \
    -d /app/ComfyUI/models/loras -o "walking_to_viewers_Wan.safetensors" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking_from_behind.safetensors" \
    -d /app/ComfyUI/models/loras && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/b3ll13-d8nc3r.safetensors" \
    -d /app/ComfyUI/models/loras && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Remade-AI/Rotate/resolve/main/rotate_20_epochs.safetensors" \
    -d /app/ComfyUI/models/loras

# Copy application files
COPY wan_handler.py /app/wan_handler.py
COPY wan22_lightx2v.py /app/wan22_lightx2v.py

# Environment variables
ENV PYTHONPATH="/app:/app/ComfyUI:/app/Practical-RIFE"

# Final verification
RUN python -c "import torch, torchvision, transformers, diffusers, accelerate, xformers; print('✅ Core packages OK')" && \
    python -c "import runpod, minio; print('✅ RunPod/MinIO OK')" || echo "⚠️ Some packages missing" && \
    python -c "import sageattention; print('✅ SageAttention OK')" || echo "⚠️ SageAttention not available"

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('🚀 Ready')" || exit 1

# Expose port
EXPOSE 8000

# Run handler
CMD ["python", "wan_handler.py"]
