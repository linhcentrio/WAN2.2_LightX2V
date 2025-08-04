# WAN2.2 LightX2V Docker Image - FIXED MODEL DOWNLOAD
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

# Install PyTorch ecosystem 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -U \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    xformers \
    --index-url https://download.pytorch.org/whl/cu126

# Verify PyTorch
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Install Triton and SageAttention
RUN pip install --no-cache-dir triton==3.2.0 && \
    pip install --no-cache-dir sageattention || \
    echo "⚠️ SageAttention installation failed, continuing without it"

# Copy requirements và install dependencies
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Clone ComfyUI và custom nodes
RUN git clone --branch ComfyUI_v0.3.47 https://github.com/Isi-dev/ComfyUI /app/ComfyUI && \
    cd /app/ComfyUI/custom_nodes && \
    git clone https://github.com/Isi-dev/ComfyUI_GGUF.git && \
    git clone --branch kjnv1.1.3 https://github.com/Isi-dev/ComfyUI_KJNodes.git

# Install custom nodes requirements
RUN cd /app/ComfyUI/custom_nodes/ComfyUI_GGUF && \
    pip install --no-cache-dir -r requirements.txt || echo "⚠️ GGUF requirements failed" && \
    cd /app/ComfyUI/custom_nodes/ComfyUI_KJNodes && \
    pip install --no-cache-dir -r requirements.txt || echo "⚠️ KJNodes requirements failed"

# Setup Practical-RIFE
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

# Create directories
RUN mkdir -p /app/ComfyUI/models/{diffusion_models,text_encoders,vae,clip_vision,loras} && \
    mkdir -p /app/ComfyUI/{input,output,temp}

# ===== FIXED MODEL DOWNLOAD SECTION =====
# Download Q6_K Models với proper filename
RUN echo "=== Downloading Q6_K Models với proper filenames ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf" \
    -d /app/ComfyUI/models/diffusion_models \
    -o wan2.2_i2v_high_noise_14B_Q6_K.gguf && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf" \
    -d /app/ComfyUI/models/diffusion_models \
    -o wan2.2_i2v_low_noise_14B_Q6_K.gguf && \
    echo "✅ Q6_K Models downloaded với proper names"

# Download supporting models với proper filename
RUN echo "=== Downloading Supporting Models với proper filenames ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    -d /app/ComfyUI/models/text_encoders \
    -o umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    -d /app/ComfyUI/models/vae \
    -o wan_2.1_vae.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    -d /app/ComfyUI/models/clip_vision \
    -o clip_vision_h.safetensors && \
    echo "✅ Supporting models downloaded với proper names"

# Download LightX2V LoRAs với proper filename
RUN echo "=== Downloading LightX2V LoRAs với proper filenames ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors && \
    echo "✅ LightX2V LoRAs downloaded với proper names"

# Download built-in LoRAs với proper filename
RUN echo "=== Downloading Built-in LoRAs với proper filenames ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking%20to%20viewers_Wan.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o "walking to viewers_Wan.safetensors" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking_from_behind.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o walking_from_behind.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/b3ll13-d8nc3r.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o b3ll13-d8nc3r.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Remade-AI/Rotate/resolve/main/rotate_20_epochs.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o rotate_20_epochs.safetensors && \
    echo "✅ Built-in LoRAs downloaded với proper names"

# IMPROVED VERIFICATION với proper filenames
RUN echo "=== FINAL VERIFICATION với proper filenames ===" && \
    echo "Checking DIT models..." && \
    test -f /app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf && echo "✅ High noise model OK" || echo "❌ High noise model MISSING" && \
    test -f /app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf && echo "✅ Low noise model OK" || echo "❌ Low noise model MISSING" && \
    echo "Checking supporting models..." && \
    test -f /app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && echo "✅ Text encoder OK" || echo "❌ Text encoder MISSING" && \
    test -f /app/ComfyUI/models/vae/wan_2.1_vae.safetensors && echo "✅ VAE OK" || echo "❌ VAE MISSING" && \
    test -f /app/ComfyUI/models/clip_vision/clip_vision_h.safetensors && echo "✅ CLIP Vision OK" || echo "❌ CLIP Vision MISSING" && \
    echo "Checking LightX2V LoRAs..." && \
    test -f /app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors && echo "✅ LightX2V rank32 OK" || echo "❌ LightX2V rank32 MISSING" && \
    test -f /app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors && echo "✅ LightX2V rank64 OK" || echo "❌ LightX2V rank64 MISSING" && \
    test -f /app/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors && echo "✅ LightX2V rank128 OK" || echo "❌ LightX2V rank128 MISSING" && \
    echo "Checking built-in LoRAs..." && \
    test -f "/app/ComfyUI/models/loras/walking to viewers_Wan.safetensors" && echo "✅ Walking to viewers OK" || echo "❌ Walking to viewers MISSING" && \
    test -f /app/ComfyUI/models/loras/walking_from_behind.safetensors && echo "✅ Walking from behind OK" || echo "❌ Walking from behind MISSING" && \
    test -f /app/ComfyUI/models/loras/b3ll13-d8nc3r.safetensors && echo "✅ Dancing LoRA OK" || echo "❌ Dancing LoRA MISSING" && \
    test -f /app/ComfyUI/models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors && echo "✅ PUSA LoRA OK" || echo "❌ PUSA LoRA MISSING" && \
    test -f /app/ComfyUI/models/loras/rotate_20_epochs.safetensors && echo "✅ Rotate LoRA OK" || echo "❌ Rotate LoRA MISSING" && \
    echo "=== DETAILED FILE LISTING ===" && \
    echo "DIT Models:" && ls -la /app/ComfyUI/models/diffusion_models/ && \
    echo "Text Encoders:" && ls -la /app/ComfyUI/models/text_encoders/ && \
    echo "VAE:" && ls -la /app/ComfyUI/models/vae/ && \
    echo "CLIP Vision:" && ls -la /app/ComfyUI/models/clip_vision/ && \
    echo "LoRAs:" && ls -la /app/ComfyUI/models/loras/ && \
    echo "=== VERIFICATION COMPLETE ==="

# Copy application files
COPY wan_handler.py /app/wan_handler.py

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
