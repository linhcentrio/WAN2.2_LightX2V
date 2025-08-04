#!/bin/bash
# build_wan22_debug.sh

echo "🏗️ Building WAN2.2 with debugging..."

# Build với verbose output
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Build và log output
docker build \
    --no-cache \
    --progress=plain \
    -t wan22-lightx2v-fixed:latest \
    -f Dockerfile . 2>&1 | tee build.log

echo "✅ Build completed. Checking models..."

# Test model verification
docker run --rm wan22-lightx2v-fixed:latest \
    python -c "
import os
models = {
    'dit_high': '/app/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_Q6_K.gguf',
    'dit_low': '/app/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_Q6_K.gguf',
    'text_encoder': '/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors',
    'vae': '/app/ComfyUI/models/vae/wan_2.1_vae.safetensors',
    'clip_vision': '/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors'
}

print('🔍 Model verification:')
for name, path in models.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f'✅ {name}: {size:.1f}MB')
    else:
        print(f'❌ {name}: MISSING')
"
