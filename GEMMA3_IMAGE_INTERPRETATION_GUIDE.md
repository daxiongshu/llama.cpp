# Gemma3 Image Interpretation Guide

## Executive Summary

`llama-gemma3-cli` is **deprecated** and has been replaced by the unified `llama-mtmd-cli` (multimodal CLI) tool. This guide explains how Gemma3 models work for image interpretation and the most efficient ways to use them in llama.cpp.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture & Image Processing Pipeline](#architecture--image-processing-pipeline)
4. [Most Efficient Usage Patterns](#most-efficient-usage-patterns)
5. [Implementation Details](#implementation-details)
6. [Performance Optimization](#performance-optimization)
7. [Advanced Usage](#advanced-usage)
8. [File Reference](#file-reference)

---

## Overview

### What is Gemma3?

Gemma3 is Google's multimodal language model family that combines vision and language understanding. The vision-capable variants (4B, 12B, 27B) use a SigLIP Vision Transformer to encode images into embeddings that the language model can process.

### Key Features

- **Vision Transformer**: SigLIP-based encoder processes 384x384 images
- **Efficient Pooling**: 4x4 spatial pooling reduces 576 patches to 36 tokens per image
- **Non-Causal Attention**: Special attention masking for image embeddings
- **Sliding Window Context**: 4096 token window with 131K total context length
- **Dynamic Resolution**: Supports variable image token counts (configurable)

### Current Status

- ✅ **Use**: `llama-mtmd-cli` (unified multimodal CLI)
- ❌ **Deprecated**: `llama-gemma3-cli` (shows deprecation warning)
- 📦 **Library**: `libmtmd` (replaces `llava.cpp`)

---

## Quick Start

### Installation

```bash
# Build the multimodal CLI
cmake -B build
cmake --build build --target llama-mtmd-cli

# Or install via Homebrew (MacOS)
brew install llama.cpp
```

### Using Pre-Quantized Models

The fastest way to get started:

```bash
# Gemma 3 4B (smallest vision model)
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF --image image.jpg -p "Describe this image"

# Gemma 3 12B (balanced)
llama-mtmd-cli -hf ggml-org/gemma-3-12b-it-GGUF --image image.jpg -p "What's in this photo?"

# Gemma 3 27B (highest quality)
llama-mtmd-cli -hf ggml-org/gemma-3-27b-it-GGUF --image image.jpg -p "Analyze this image"
```

**Note**: The 1B model does NOT support vision.

### Converting Your Own Models

```bash
# Download model from Hugging Face
git clone https://huggingface.co/google/gemma-3-4b-it
cd gemma-3-4b-it

# Convert with vision projector
python ../llama.cpp/convert_hf_to_gguf.py --outfile model.gguf --outtype f16 --mmproj .

# This creates:
# - model.gguf (text model)
# - mmproj-model.gguf (vision projector)
```

### Running with Converted Models

```bash
./build/bin/llama-mtmd-cli \
  -m model.gguf \
  --mmproj mmproj-model.gguf \
  --image photo.jpg \
  -p "Describe this image in detail"
```

---

## Architecture & Image Processing Pipeline

### Complete Pipeline Overview

```
Input Image (JPG/PNG/BMP)
    ↓
[1] Load & Decode (stb_image)
    → RGB uint8 array
    ↓
[2] Preprocessing
    → Resize to 384x384
    → Normalize to float32 [-1, 1]
    ↓
[3] Vision Transformer (SigLIP)
    → Patch embedding (16x16 patches)
    → 576 patches (24x24 grid)
    → Each patch: 1152-dim vector
    ↓
[4] Spatial Pooling (Gemma3-specific)
    → 4x4 average pooling
    → Reduces to 36 tokens (6x6 grid)
    ↓
[5] RMS Normalization
    → Apply soft_emb_norm weight
    ↓
[6] Projection
    → Linear transformation (1152 → n_embd)
    → Produces final image embeddings
    ↓
[7] Token Wrapping
    → Add <start_of_image> token
    → Insert 36 image embeddings
    → Add <end_of_image> token
    ↓
[8] Language Model Processing
    → Non-causal attention for image tokens
    → Standard causal attention for text
    → Generate response
```

### Key Components

#### Vision Encoder: SigLIP

- **Architecture**: Vision Transformer (ViT)
- **Input Size**: 384×384 pixels
- **Patch Size**: 16×16 pixels
- **Number of Patches**: 24×24 = 576
- **Embedding Dim**: 1152

**File**: `tools/mtmd/clip.cpp:558-579`

#### Gemma3 Vision Projector

The projector reduces 576 patch embeddings to 36 image tokens:

```cpp
// Transpose to [n_embd, n_patches]
cur = ggml_transpose(ctx0, cur);

// Reshape to spatial 4D tensor [24, 24, 1152, 1]
cur = ggml_cont_4d(ctx0, cur, 24, 24, n_embd, 1);

// 4x4 average pooling → [6, 6, 1152, 1]
cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, 4, 4, 4, 4, 0, 0);

// Reshape to [36, 1152]
cur = ggml_reshape_3d(ctx0, cur, 36, n_embd, 1);

// RMS normalization
cur = ggml_rms_norm(ctx0, cur, eps);
cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

// Final projection to language model dimension
cur = ggml_mul_mat(ctx0,
    ggml_transpose(ctx0, model.mm_input_proj_w),
    cur);
```

**Key Tensors**:
- `mm.input_projection.weight` - Projection matrix [n_embd, 1152]
- `mm.soft_emb_norm.weight` - RMS normalization scaling

#### Text Model Architecture

**File**: `src/models/gemma3-iswa.cpp`

- **Type**: GEMMA3 (Interleaved Sliding Window Attention)
- **Context**: 131,072 tokens total
- **Sliding Window**: 4,096 tokens
- **RoPE Theta**: 1,000,000
- **Attention Scale**: Custom normalization factor
- **Special Feature**: No weight scaling for raw embeddings (images)

```cpp
// Line 12-14: Images bypass the usual input scaling
if (ubatch.token) {
    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
}
```

---

## Most Efficient Usage Patterns

### 1. Single Image Query (Fastest)

**Best for**: Quick one-off image analysis

```bash
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF \
  --image photo.jpg \
  -p "What objects are in this image?" \
  -n 256  # Limit output tokens
```

**Efficiency Tips**:
- Use 4B model for fastest inference
- Limit `-n` (max tokens) to needed length
- Use GPU offloading: `--no-mmproj-offload` to disable if needed

### 2. Multiple Images in One Prompt

**Best for**: Comparing images or multi-image analysis

```bash
llama-mtmd-cli -hf ggml-org/gemma-3-12b-it-GGUF \
  --image img1.jpg --image img2.jpg \
  -p "<__media__> vs <__media__>: What are the differences?"
```

**Key Point**: Use `<__media__>` marker for each image. The default marker is returned by `mtmd_default_marker()`.

### 3. Interactive Chat Mode (Most Flexible)

**Best for**: Exploratory analysis, multiple queries on same images

```bash
# Start without -p flag to enter chat mode
llama-mtmd-cli -hf ggml-org/gemma-3-12b-it-GGUF

# Interactive commands:
> /image photo.jpg          # Load an image
> What's in this image?     # Ask questions
> /image another.jpg        # Load another image
> Compare these two images  # Analyze both
> /clear                    # Clear history
> /exit                     # Quit
```

**Advantages**:
- Reuse KV cache across queries
- Build context with multiple images
- Lower latency for follow-up questions

**File**: `tools/mtmd/mtmd-cli.cpp:334-401`

### 4. Batch Processing (Custom Integration)

**Best for**: Processing many images programmatically

Use the C API from `mtmd.h`:

```c
#include "mtmd.h"

// Initialize context
mtmd_context_params params = mtmd_context_params_default();
params.use_gpu = true;
params.n_threads = 8;
mtmd_context* ctx = mtmd_init_from_file("mmproj.gguf", model, params);

// Load and process image
mtmd_bitmap* bitmap = mtmd_bitmap_init(width, height, rgb_data);
mtmd_input_chunks* chunks = mtmd_input_chunks_init();

// Tokenize with image
mtmd_input_text text = {
    .text = "Describe: <__media__>",
    .add_special = true,
    .parse_special = true
};
mtmd_tokenize(ctx, chunks, &text, &bitmap, 1);

// Process chunks
for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
    mtmd_encode_chunk(ctx, chunk);
    float* embeddings = mtmd_get_output_embd(ctx);
    // Feed embeddings to language model
}

// Cleanup
mtmd_input_chunks_free(chunks);
mtmd_bitmap_free(bitmap);
mtmd_free(ctx);
```

**File**: `tools/mtmd/mtmd.h`

### 5. Dynamic Resolution Control

**Best for**: Balancing quality vs speed

```bash
# Minimum tokens (faster, lower quality)
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF \
  --image-min-tokens 36 \
  --image-max-tokens 36 \
  --image photo.jpg

# Maximum tokens (slower, higher quality)
llama-mtmd-cli -hf ggml-org/gemma-3-12b-it-GGUF \
  --image-min-tokens 36 \
  --image-max-tokens 144 \
  --image photo.jpg
```

**Default**: 36 tokens (6×6 grid after 4×4 pooling)

**Parameters**:
- `--image-min-tokens`: Minimum image tokens (default: from metadata)
- `--image-max-tokens`: Maximum image tokens (default: from metadata)

**File**: `tools/mtmd/mtmd-cli.cpp:140-141`

---

## Implementation Details

### Image Preprocessing

**File**: `tools/mtmd/clip.cpp:4263-4273`

1. **Load**: Uses `stb_image` to load JPG/PNG/BMP
2. **Resize**: Square resize to 384×384
3. **Normalize**: Scale to [-1, 1] range
4. **Format**: RGB uint8 → float32

### Tokenization Process

**File**: `tools/mtmd/mtmd-cli.cpp:223-267`

```cpp
// Create input with media marker
mtmd_input_text text = {
    .text = formatted_chat.c_str(),
    .add_special = add_bos,
    .parse_special = true
};

// Tokenize with images
mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(),
              &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());

// Eval chunks
mtmd_helper_eval_chunks(ctx_vision.get(), lctx,
                        chunks.ptr.get(), n_past, 0,
                        n_batch, true, &new_n_past);
```

### Non-Causal Attention

Gemma3 requires special attention masking for image tokens:

- **Image tokens**: Full bi-directional attention
- **Text tokens**: Standard causal (left-to-right) attention

**Check**: `mtmd_decode_use_non_causal(ctx)` returns `true` for Gemma3

**File**: `tools/mtmd/mtmd.h:105`

### Special Tokens

- `<start_of_image>`: Marks beginning of image embeddings
- `<end_of_image>`: Marks end of image embeddings
- `<__media__>`: Placeholder in prompts (replaced with image)

**File**: `tools/mtmd/mtmd.h:42`, `mtmd.c:92`

---

## Performance Optimization

### 1. Model Size Selection

| Model | Parameters | VRAM (FP16) | Speed | Quality | Use Case |
|-------|-----------|-------------|-------|---------|----------|
| 4B    | 4 billion | ~8 GB       | ⚡⚡⚡   | ⭐⭐     | Real-time, demos |
| 12B   | 12 billion| ~24 GB      | ⚡⚡     | ⭐⭐⭐   | General use |
| 27B   | 27 billion| ~54 GB      | ⚡       | ⭐⭐⭐⭐ | High quality |

### 2. Quantization

Convert to lower precision for faster inference:

```bash
# Q4_K_M: Good balance (4-bit)
./build/bin/llama-quantize model.gguf model-q4km.gguf Q4_K_M

# Q8_0: Higher quality (8-bit)
./build/bin/llama-quantize model.gguf model-q8.gguf Q8_0
```

**Important**: The vision projector (`mmproj`) is kept in F16 for accuracy.

### 3. GPU Offloading

```bash
# Offload all layers to GPU
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF \
  --image img.jpg \
  -ngl 999  # Offload all layers

# Disable mmproj GPU usage (if memory constrained)
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF \
  --image img.jpg \
  --no-mmproj-offload
```

### 4. Thread Optimization

```bash
# Set CPU threads
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF \
  --image img.jpg \
  -t 8  # Use 8 CPU threads
```

**Recommendation**: Set to number of physical cores, not hyperthreads.

### 5. Batch Size Tuning

```bash
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF \
  --image img.jpg \
  -b 512  # Batch size for prompt processing
```

**Default**: 2048

**Trade-off**: Larger = faster prompt processing, more VRAM

### 6. Flash Attention

```bash
llama-mtmd-cli -hf ggml-org/gemma-3-12b-it-GGUF \
  --image img.jpg \
  -fa  # Enable flash attention
```

**Requirement**: CUDA-capable GPU with compute capability 7.0+

**File**: `tools/mtmd/mtmd-cli.cpp:139`

### 7. Caching Strategy

**In chat mode**: KV cache is preserved between turns
- First image: ~2-3s processing
- Follow-up questions: ~0.1-0.5s (cached)

**To clear cache**: Use `/clear` command in chat mode

**File**: `tools/mtmd/mtmd-cli.cpp:363-368`

---

## Advanced Usage

### Custom Chat Templates

Some older models need specific templates:

```bash
# LLaVA models
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf \
  --chat-template vicuna

# MobileVLM models
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf \
  --chat-template deepseek

# Mistral Small 3.1
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf \
  --chat-template mistral-v7
```

**File**: `tools/mtmd/mtmd-cli.cpp:104-110`

### Server Integration

For production use, integrate with the server API:

```bash
# Start server with multimodal support
./build/bin/llama-server \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --host 0.0.0.0 \
  --port 8080

# Send request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }'
```

**Tests**: `tools/server/tests/unit/test_vision_api.py`

### Custom Merge Factor

The `n_merge` parameter controls pooling (default: 4):

```python
# During conversion, customize merge factor
python convert_hf_to_gguf.py \
  --outfile model.gguf \
  --mmproj . \
  --n-merge 2  # Use 2x2 pooling instead of 4x4
```

**Result**: More image tokens (144 instead of 36), higher quality, slower

**File**: `tools/mtmd/clip.cpp:562`

### Audio Support

`libmtmd` also supports audio (though not in Gemma3):

```c
// Load audio (PCM F32)
mtmd_bitmap* audio = mtmd_bitmap_init_from_audio(n_samples, float_data);

// Check support
if (mtmd_support_audio(ctx)) {
    int bitrate = mtmd_get_audio_bitrate(ctx);  // e.g., 16000 Hz
}
```

**File**: `tools/mtmd/mtmd.h:111-118`

---

## File Reference

### Core Implementation

| File | Purpose | Lines | Key Functions |
|------|---------|-------|---------------|
| `tools/mtmd/mtmd-cli.cpp` | Main CLI entry point | 407 | `main()`, `eval_message()`, `generate_response()` |
| `tools/mtmd/mtmd.h` | C API definitions | 304 | `mtmd_tokenize()`, `mtmd_encode_chunk()` |
| `tools/mtmd/mtmd.cpp` | Library implementation | ~2000+ | Core processing logic |
| `tools/mtmd/clip.cpp` | Vision encoder | ~5000+ | SigLIP, projector types |
| `tools/mtmd/clip.h` | Vision encoder API | ~200 | Encoder interface |
| `src/models/gemma3-iswa.cpp` | Text model architecture | 132 | `llm_build_gemma3_iswa()` |

### Documentation

| File | Purpose |
|------|---------|
| `docs/multimodal/gemma3.md` | Gemma3 quick start guide |
| `tools/mtmd/README.md` | Multimodal overview & history |

### Model Definitions

| File | Purpose |
|------|---------|
| `src/llama-arch.h` | Architecture constants |
| `src/llama-hparams.h` | Hyperparameter definitions |
| `gguf-py/gguf/constants.py` | GGUF format constants |

### Key Code Sections

- **Gemma3 Projector**: `tools/mtmd/clip.cpp:558-579`
- **Image Preprocessing**: `tools/mtmd/clip.cpp:4263-4273`
- **Tokenization**: `tools/mtmd/mtmd-cli.cpp:223-267`
- **Chat Mode**: `tools/mtmd/mtmd-cli.cpp:334-401`
- **Projector Loading**: `tools/mtmd/clip.cpp:2777-2783`, `3112-3116`
- **Non-causal Attention**: `src/models/gemma3-iswa.cpp:12-14`

---

## Troubleshooting

### Common Issues

**Issue**: "Model does not have chat template"
```bash
# Solution: Specify template explicitly
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf \
  --chat-template vicuna \
  --image img.jpg
```

**Issue**: Out of memory
```bash
# Solution: Use quantized model + disable mmproj offload
llama-mtmd-cli -m model-q4km.gguf --mmproj mmproj.gguf \
  --no-mmproj-offload \
  --image img.jpg
```

**Issue**: Slow inference
```bash
# Solution: Enable GPU offloading + flash attention
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF \
  -ngl 999 -fa \
  --image img.jpg
```

---

## Summary: Best Practices

1. ✅ **Use `llama-mtmd-cli`** (not deprecated `llama-gemma3-cli`)
2. ✅ **Start with 4B model** for fastest results
3. ✅ **Use pre-quantized models** from `ggml-org` HF account
4. ✅ **Enable GPU offloading** with `-ngl 999`
5. ✅ **Use chat mode** for multiple queries on same images
6. ✅ **Limit output tokens** with `-n` for faster responses
7. ✅ **Keep mmproj in F16** for best quality
8. ✅ **Use flash attention** on supported GPUs
9. ✅ **Cache prompts** by staying in chat mode
10. ✅ **Profile first**: Test speed before scaling

---

## Additional Resources

- **Hugging Face**: https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d
- **llama.cpp Docs**: https://github.com/ggerganov/llama.cpp
- **Issues**: https://github.com/ggerganov/llama.cpp/issues
- **Pre-quantized Models**: https://huggingface.co/ggml-org

---

*Guide generated on 2025-11-11 for llama.cpp commit `48bd265`*
