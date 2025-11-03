# llama-server CLI Documentation

Comprehensive documentation for the `llama-server` command-line interface. This document details all available parameters with their meanings, default values, and links to source code implementation.

## Table of Contents

- [Server Configuration](#server-configuration)
- [Model Loading](#model-loading)
- [Context and Memory](#context-and-memory)
- [Performance and Hardware](#performance-and-hardware)
- [Chat and Templates](#chat-and-templates)
- [Server Endpoints](#server-endpoints)
- [Speculative Decoding](#speculative-decoding)
- [Sampling Parameters](#sampling-parameters)
- [Advanced Options](#advanced-options)
- [Preset Configurations](#preset-configurations)
- [Environment Variables](#environment-variables)

---

## Server Configuration

### `--host HOST`
IP address to listen on, or bind to a UNIX socket if the address ends with `.sock`
- **Default:** `127.0.0.1`
- **Environment Variable:** `LLAMA_ARG_HOST`
- **Source:** [common/arg.cpp:3285-3290](common/arg.cpp#L3285-L3290)

### `--port PORT`
Port to listen on
- **Default:** `8080`
- **Environment Variable:** `LLAMA_ARG_PORT`
- **Source:** [common/arg.cpp:3292-3297](common/arg.cpp#L3292-L3297)

### `--path PATH`
Path to serve static files from (for Web UI)
- **Default:** `""`
- **Environment Variable:** `LLAMA_ARG_STATIC_PATH`
- **Source:** [common/arg.cpp:3299-3304](common/arg.cpp#L3299-L3304)

### `--api-prefix PREFIX`
Prefix path the server serves from, without the trailing slash
- **Default:** `""`
- **Environment Variable:** `LLAMA_ARG_API_PREFIX`
- **Source:** [common/arg.cpp:3306-3311](common/arg.cpp#L3306-L3311)

### `--api-key KEY`
API key to use for authentication (can be specified multiple times)
- **Default:** none
- **Environment Variable:** `LLAMA_API_KEY`
- **Source:** [common/arg.cpp:3335-3340](common/arg.cpp#L3335-L3340)

### `--api-key-file FNAME`
Path to file containing API keys (one per line)
- **Default:** none
- **Source:** [common/arg.cpp:3342-3357](common/arg.cpp#L3342-L3357)
- **Implementation:** Reads file line by line and adds each non-empty line as an API key

### `--ssl-key-file FNAME`
Path to file containing a PEM-encoded SSL private key
- **Default:** `""`
- **Environment Variable:** `LLAMA_ARG_SSL_KEY_FILE`
- **Source:** [common/arg.cpp:3359-3364](common/arg.cpp#L3359-L3364)

### `--ssl-cert-file FNAME`
Path to file containing a PEM-encoded SSL certificate
- **Default:** `""`
- **Environment Variable:** `LLAMA_ARG_SSL_CERT_FILE`
- **Source:** [common/arg.cpp:3366-3371](common/arg.cpp#L3366-L3371)

### `-to, --timeout N`
Server read/write timeout in seconds
- **Default:** `600`
- **Environment Variable:** `LLAMA_ARG_TIMEOUT`
- **Source:** [common/arg.cpp:3383-3389](common/arg.cpp#L3383-L3389)
- **Note:** Sets both `timeout_read` and `timeout_write`

### `--threads-http N`
Number of threads used to process HTTP requests
- **Default:** `-1` (auto)
- **Environment Variable:** `LLAMA_ARG_THREADS_HTTP`
- **Source:** [common/arg.cpp:3391-3396](common/arg.cpp#L3391-L3396)

---

## Model Loading

### `-m, --model FNAME`
Model path
- **Default:** `models/$filename` (from `--hf-file` or `--model-url` if set)
- **Environment Variable:** `LLAMA_ARG_MODEL`
- **Source:** [common/arg.cpp:3049-3059](common/arg.cpp#L3049-L3059)

### `-mu, --model-url MODEL_URL`
Model download URL
- **Default:** unused
- **Environment Variable:** `LLAMA_ARG_MODEL_URL`
- **Source:** [common/arg.cpp:3061-3066](common/arg.cpp#L3061-L3066)

### `-a, --alias STRING`
Set alias for model name (to be used by REST API)
- **Default:** `""`
- **Environment Variable:** `LLAMA_ARG_ALIAS`
- **Source:** [common/arg.cpp:3042-3047](common/arg.cpp#L3042-L3047)

### `--pooling {none,mean,cls,last,rank}`
Pooling type for embeddings
- **Default:** model default
- **Environment Variable:** `LLAMA_ARG_POOLING`
- **Source:** [common/arg.cpp:2491](common/arg.cpp#L2491)
- **Options:**
  - `none`: No pooling
  - `mean`: Mean pooling
  - `cls`: CLS token pooling
  - `last`: Last token pooling
  - `rank`: Rank pooling (used for reranking)

---

## Context and Memory

### `-c, --ctx-size N`
Size of the prompt context
- **Default:** `0` (loaded from model)
- **Environment Variable:** `LLAMA_ARG_CTX_SIZE`
- **Source:** [common/arg.cpp:1884-1889](common/arg.cpp#L1884-L1889)

### `-b, --batch-size N`
Logical maximum batch size
- **Default:** `2048`
- **Environment Variable:** `LLAMA_ARG_BATCH`
- **Source:** [common/arg.cpp:1902-1907](common/arg.cpp#L1902-L1907)

### `-ub, --ubatch-size N`
Physical maximum batch size
- **Default:** `512`
- **Environment Variable:** `LLAMA_ARG_UBATCH`
- **Source:** [common/arg.cpp:1909-1914](common/arg.cpp#L1909-L1914)

### `--ctx-checkpoints, --swa-checkpoints N`
Maximum number of context checkpoints to create per slot
- **Default:** `8`
- **Environment Variable:** `LLAMA_ARG_CTX_CHECKPOINTS`
- **Source:** [common/arg.cpp:1931-1937](common/arg.cpp#L1931-L1937)
- **More info:** [GitHub PR #15293](https://github.com/ggml-org/llama.cpp/pull/15293)

### `--cache-ram, -cram N`
Set the maximum cache size in MiB
- **Default:** `8192`
- **Values:** `-1` = no limit, `0` = disable
- **Environment Variable:** `LLAMA_ARG_CACHE_RAM`
- **Source:** [common/arg.cpp:1939-1945](common/arg.cpp#L1939-L1945)
- **More info:** [GitHub PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391)

### `--cache-reuse N`
Minimum chunk size to attempt reusing from the cache via KV shifting
- **Default:** `0`
- **Environment Variable:** `LLAMA_ARG_CACHE_REUSE`
- **Source:** [common/arg.cpp:3398-3406](common/arg.cpp#L3398-L3406)
- **More info:** [GitHub visualization](https://ggml.ai/f0.png)

### `-ctk, --cache-type-k TYPE`
KV cache data type for K
- **Default:** `f16`
- **Environment Variable:** `LLAMA_ARG_CACHE_TYPE_K`
- **Source:** [common/arg.cpp:2603-2614](common/arg.cpp#L2603-L2614)
- **Allowed values:** See [KV cache types](#kv-cache-types)

### `-ctv, --cache-type-v TYPE`
KV cache data type for V
- **Default:** `f16`
- **Environment Variable:** `LLAMA_ARG_CACHE_TYPE_V`
- **Source:** [common/arg.cpp:2616-2627](common/arg.cpp#L2616-L2627)
- **Allowed values:** See [KV cache types](#kv-cache-types)

### `--no-context-shift`
Disables context shift on infinite text generation
- **Default:** context shift is enabled
- **Environment Variable:** `LLAMA_ARG_NO_CONTEXT_SHIFT`
- **Source:** [common/arg.cpp:1955-1960](common/arg.cpp#L1955-L1960)

### `--context-shift`
Enables context shift on infinite text generation
- **Default:** enabled
- **Environment Variable:** `LLAMA_ARG_CONTEXT_SHIFT`
- **Source:** [common/arg.cpp:1962-1967](common/arg.cpp#L1962-L1967)

### `--slot-save-path PATH`
Path to save slot KV cache
- **Default:** disabled
- **Source:** [common/arg.cpp:3436-3445](common/arg.cpp#L3436-L3445)
- **Note:** Automatically adds directory separator if not present

---

## Performance and Hardware

### `-t, --threads N`
Number of CPU threads to use during generation
- **Default:** hardware concurrency
- **Environment Variable:** `LLAMA_ARG_THREADS`
- **Source:** [common/arg.cpp:1762-1770](common/arg.cpp#L1762-L1770)

### `-tb, --threads-batch N`
Number of threads to use during batch and prompt processing
- **Default:** same as `--threads`
- **Source:** [common/arg.cpp:1772-1780](common/arg.cpp#L1772-L1780)

### `-ngl, --gpu-layers, --n-gpu-layers N`
Number of layers to store in VRAM
- **Default:** `0`
- **Environment Variable:** `LLAMA_ARG_N_GPU_LAYERS`
- **Source:** common/arg.cpp (GPU layer parameter)
- **Note:** Warns if no GPU support is compiled

### `-fa, --flash-attn [on|off|auto]`
Set Flash Attention use
- **Default:** `auto`
- **Environment Variable:** `LLAMA_ARG_FLASH_ATTN`
- **Source:** [common/arg.cpp:1975-1989](common/arg.cpp#L1975-L1989)
- **Options:**
  - `on`/`enabled`/`1`: Force enable
  - `off`/`disabled`/`0`: Force disable
  - `auto`/`-1`: Auto-detect

### `-np, --parallel N`
Number of parallel sequences to decode
- **Default:** `1`
- **Environment Variable:** `LLAMA_ARG_N_PARALLEL`
- **Source:** [common/arg.cpp:2708-2713](common/arg.cpp#L2708-L2713)

### `-cb, --cont-batching`
Enable continuous batching (a.k.a dynamic batching)
- **Default:** disabled
- **Environment Variable:** `LLAMA_ARG_CONT_BATCHING`
- **Source:** [common/arg.cpp:2722-2727](common/arg.cpp#L2722-L2727)

### `-nocb, --no-cont-batching`
Disable continuous batching
- **Environment Variable:** `LLAMA_ARG_NO_CONT_BATCHING`
- **Source:** [common/arg.cpp:2729-2734](common/arg.cpp#L2729-L2734)

### `--no-warmup`
Skip warming up the model with an empty run
- **Default:** warmup enabled
- **Source:** [common/arg.cpp:2189-2194](common/arg.cpp#L2189-L2194)

---

## Chat and Templates

### `--chat-template JINJA_TEMPLATE`
Set custom Jinja chat template
- **Default:** template taken from model's metadata
- **Environment Variable:** `LLAMA_ARG_CHAT_TEMPLATE`
- **Source:** [common/arg.cpp:3473-3483](common/arg.cpp#L3473-L3483)
- **Note:** If suffix/prefix are specified, template will be disabled

### `--chat-template-file JINJA_TEMPLATE_FILE`
Set custom Jinja chat template from file
- **Default:** template taken from model's metadata
- **Environment Variable:** `LLAMA_ARG_CHAT_TEMPLATE_FILE`
- **Source:** [common/arg.cpp:3485-3495](common/arg.cpp#L3485-L3495)

### `--chat-template-kwargs STRING`
Sets additional params for the JSON template parser (JSON format)
- **Default:** `{}`
- **Environment Variable:** `LLAMA_CHAT_TEMPLATE_KWARGS`
- **Source:** [common/arg.cpp:3373-3381](common/arg.cpp#L3373-L3381)
- **Example:** `--chat-template-kwargs '{"key": "value"}'`

### `--jinja`
Use Jinja template for chat
- **Default:** disabled
- **Environment Variable:** `LLAMA_ARG_JINJA`
- **Source:** [common/arg.cpp:3447-3452](common/arg.cpp#L3447-L3452)

### `--reasoning-format FORMAT`
Controls whether thought tags are allowed and/or extracted from the response
- **Default:** `auto`
- **Environment Variable:** `LLAMA_ARG_THINK`
- **Source:** [common/arg.cpp:3454-3463](common/arg.cpp#L3454-L3463)
- **Options:**
  - `none`: Leaves thoughts unparsed in `message.content`
  - `deepseek`: Puts thoughts in `message.reasoning_content`
  - `deepseek-legacy`: Keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`

### `--reasoning-budget N`
Controls the amount of thinking allowed
- **Default:** `-1` (unrestricted)
- **Environment Variable:** `LLAMA_ARG_THINK_BUDGET`
- **Source:** [common/arg.cpp:3465-3471](common/arg.cpp#L3465-L3471)
- **Options:**
  - `-1`: Unrestricted thinking budget
  - `0`: Disable thinking

### `--no-prefill-assistant`
Don't prefill the assistant's response if the last message is an assistant message
- **Default:** prefill enabled
- **Environment Variable:** `LLAMA_ARG_NO_PREFILL_ASSISTANT`
- **Source:** [common/arg.cpp:3497-3505](common/arg.cpp#L3497-L3505)

---

## Server Endpoints

### `--no-webui`
Disable the Web UI
- **Default:** Web UI enabled
- **Environment Variable:** `LLAMA_ARG_NO_WEBUI`
- **Source:** [common/arg.cpp:3313-3318](common/arg.cpp#L3313-L3318)

### `--metrics`
Enable Prometheus-compatible metrics endpoint
- **Default:** disabled
- **Environment Variable:** `LLAMA_ARG_ENDPOINT_METRICS`
- **Source:** [common/arg.cpp:3408-3413](common/arg.cpp#L3408-L3413)

### `--props`
Enable changing global properties via POST /props
- **Default:** disabled
- **Environment Variable:** `LLAMA_ARG_ENDPOINT_PROPS`
- **Source:** [common/arg.cpp:3415-3420](common/arg.cpp#L3415-L3420)
- **Note:** Only controls POST requests, not GET

### `--slots`
Enable slots monitoring endpoint
- **Default:** enabled
- **Environment Variable:** `LLAMA_ARG_ENDPOINT_SLOTS`
- **Source:** [common/arg.cpp:3422-3427](common/arg.cpp#L3422-L3427)

### `--no-slots`
Disable slots monitoring endpoint
- **Environment Variable:** `LLAMA_ARG_NO_ENDPOINT_SLOTS`
- **Source:** [common/arg.cpp:3429-3434](common/arg.cpp#L3429-L3434)

### `--embedding, --embeddings`
Restrict to only support embedding use case; use only with dedicated embedding models
- **Default:** disabled
- **Environment Variable:** `LLAMA_ARG_EMBEDDINGS`
- **Source:** [common/arg.cpp:3320-3325](common/arg.cpp#L3320-L3325)

### `--reranking, --rerank`
Enable reranking endpoint on server
- **Default:** disabled
- **Environment Variable:** `LLAMA_ARG_RERANKING`
- **Source:** [common/arg.cpp:3327-3333](common/arg.cpp#L3327-L3333)
- **Note:** Also sets `embedding = true` and `pooling_type = LLAMA_POOLING_TYPE_RANK`

---

## Speculative Decoding

### `-md, --model-draft FNAME`
Draft model for speculative decoding
- **Default:** unused
- **Environment Variable:** `LLAMA_ARG_MODEL_DRAFT`
- **Source:** [common/arg.cpp:3807-3812](common/arg.cpp#L3807-L3812)

### `--draft-max, --draft, --draft-n N`
Number of tokens to draft for speculative decoding
- **Default:** `16`
- **Environment Variable:** `LLAMA_ARG_DRAFT_MAX`
- **Source:** [common/arg.cpp:3752-3757](common/arg.cpp#L3752-L3757)

### `--draft-min, --draft-n-min N`
Minimum number of draft tokens to use for speculative decoding
- **Default:** `5`
- **Environment Variable:** `LLAMA_ARG_DRAFT_MIN`
- **Source:** [common/arg.cpp:3759-3764](common/arg.cpp#L3759-L3764)

### `--draft-p-min P`
Minimum speculative decoding probability (greedy)
- **Default:** `0.9`
- **Environment Variable:** `LLAMA_ARG_DRAFT_P_MIN`
- **Source:** [common/arg.cpp:3773-3778](common/arg.cpp#L3773-L3778)

### `-cd, --ctx-size-draft N`
Size of the prompt context for the draft model
- **Default:** `0` (loaded from model)
- **Environment Variable:** `LLAMA_ARG_CTX_SIZE_DRAFT`
- **Source:** [common/arg.cpp:3780-3785](common/arg.cpp#L3780-L3785)

### `-ngld, --gpu-layers-draft, --n-gpu-layers-draft N`
Number of layers to store in VRAM for the draft model
- **Default:** `0`
- **Environment Variable:** `LLAMA_ARG_N_GPU_LAYERS_DRAFT`
- **Source:** [common/arg.cpp:3795-3805](common/arg.cpp#L3795-L3805)

### `-devd, --device-draft <dev1,dev2,..>`
Comma-separated list of devices to use for offloading the draft model
- **Default:** none
- **Source:** [common/arg.cpp:3787-3793](common/arg.cpp#L3787-L3793)
- **Note:** Use `--list-devices` to see available devices

### `--spec-replace TARGET DRAFT`
Translate the string in TARGET into DRAFT if the draft model and main model are not compatible
- **Source:** [common/arg.cpp:3814-3819](common/arg.cpp#L3814-L3819)

### `-ctkd, --cache-type-k-draft TYPE`
KV cache data type for K for the draft model
- **Default:** `f16`
- **Environment Variable:** `LLAMA_ARG_CACHE_TYPE_K_DRAFT`
- **Source:** [common/arg.cpp:3821-3832](common/arg.cpp#L3821-L3832)

### `-ctvd, --cache-type-v-draft TYPE`
KV cache data type for V for the draft model
- **Default:** `f16`
- **Environment Variable:** `LLAMA_ARG_CACHE_TYPE_V_DRAFT`
- **Source:** [common/arg.cpp:3834-3845](common/arg.cpp#L3834-L3845)

---

## Sampling Parameters

All sampling parameters are marked with `set_sparam()` in the source code. These parameters control how tokens are generated.

### `-s, --seed SEED`
RNG seed
- **Default:** `-1` (random seed)
- **Source:** [common/arg.cpp:2214-2219](common/arg.cpp#L2214-L2219)

### `--temp N`
Temperature
- **Default:** `0.8`
- **Source:** [common/arg.cpp:2235-2241](common/arg.cpp#L2235-L2241)
- **Note:** Minimum value is 0.0

### `--top-k N`
Top-k sampling
- **Default:** `40`
- **Source:** [common/arg.cpp:2243-2248](common/arg.cpp#L2243-L2248)
- **Note:** 0 = disabled

### `--top-p N`
Top-p sampling
- **Default:** `0.95`
- **Source:** common/arg.cpp (top-p parameter)

### `--samplers SAMPLERS`
Samplers that will be used for generation in order, separated by `;`
- **Default:** `top_k;tfs_z;typ_p;top_p;min_p;xtc;temperature`
- **Source:** [common/arg.cpp:2206-2212](common/arg.cpp#L2206-L2212)

### `--ignore-eos`
Ignore end of stream token and continue generating
- **Default:** false
- **Source:** [common/arg.cpp:2228-2233](common/arg.cpp#L2228-L2233)
- **Note:** Implies `--logit-bias EOS-inf`

---

## Advanced Options

### `-sps, --slot-prompt-similarity SIMILARITY`
How much the prompt of a request must match the prompt of a slot in order to use that slot
- **Default:** `0.1` (0.0 = disabled)
- **Source:** [common/arg.cpp:3507-3512](common/arg.cpp#L3507-L3512)

### `--lora-init-without-apply`
Load LoRA adapters without applying them (apply later via POST /lora-adapters)
- **Default:** disabled
- **Environment Variable:** N/A
- **Source:** [common/arg.cpp:3514-3519](common/arg.cpp#L3514-L3519)

### `--spm-infill`
Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this
- **Default:** disabled
- **Source:** [common/arg.cpp:2196-2204](common/arg.cpp#L2196-L2204)

### `-r, --reverse-prompt PROMPT`
Halt generation at PROMPT, return control in interactive mode
- **Source:** [common/arg.cpp:2104-2109](common/arg.cpp#L2104-L2109)
- **Note:** Can be specified multiple times

### `-sp, --special`
Special tokens output enabled
- **Default:** false
- **Source:** [common/arg.cpp:2111-2116](common/arg.cpp#L2111-L2116)

### `--no-perf`
Disable internal libllama performance timings
- **Default:** false
- **Environment Variable:** `LLAMA_ARG_NO_PERF`
- **Source:** [common/arg.cpp:2005-2011](common/arg.cpp#L2005-L2011)

---

## Preset Configurations

The following are convenience presets that configure multiple parameters at once for specific models:

### `--embedding-default`
Use default embedding model configuration
- **Source:** [common/arg.cpp:3966-3975](common/arg.cpp#L3966-L3975)
- **Sets:**
  - `n_parallel = 32`
  - `n_ctx = 2048 * n_parallel`
  - `verbose_prompt = true`
  - `embedding = true`

### `--fim-qwen-1.5b-default`
Use default Qwen 2.5 Coder 1.5B
- **Source:** [common/arg.cpp:3978-3989](common/arg.cpp#L3978-L3989)
- **Sets:**
  - Model: `ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF`
  - Port: `8012`
  - `n_ubatch = 1024`
  - `n_batch = 1024`
  - `n_cache_reuse = 256`

### `--fim-qwen-3b-default`
Use default Qwen 2.5 Coder 3B
- **Source:** [common/arg.cpp:3992-4003](common/arg.cpp#L3992-L4003)
- **Configuration:** Similar to 1.5B with different model file

### `--fim-qwen-7b-default`
Use default Qwen 2.5 Coder 7B
- **Source:** [common/arg.cpp:4006-4017](common/arg.cpp#L4006-L4017)
- **Configuration:** Similar to 1.5B with different model file

### `--fim-qwen-7b-spec`
Use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding
- **Source:** [common/arg.cpp:4020-4033](common/arg.cpp#L4020-L4033)
- **Sets:** Main model: 7B, Draft model: 0.5B

### `--fim-qwen-14b-spec`
Use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding
- **Source:** [common/arg.cpp:4036-4049](common/arg.cpp#L4036-L4049)
- **Sets:** Main model: 14B, Draft model: 0.5B

### `--fim-qwen-30b-default`
Use default Qwen 3 Coder 30B A3B Instruct
- **Source:** [common/arg.cpp:4052-4063](common/arg.cpp#L4052-L4063)

### `--gpt-oss-20b-default`
Use GPT-OSS-20B
- **Source:** [common/arg.cpp:4066-4083](common/arg.cpp#L4066-L4083)
- **Sets:**
  - Port: `8013`
  - `n_ubatch = 2048`
  - `n_batch = 32768`
  - `n_parallel = 2`
  - `n_ctx = 131072 * n_parallel`
  - Custom sampling parameters

### `--gpt-oss-120b-default`
Use GPT-OSS-120B
- **Source:** [common/arg.cpp:4086-4102](common/arg.cpp#L4086-L4102)

### `--vision-gemma-4b-default`
Use Gemma 3 4B QAT
- **Source:** [common/arg.cpp:4105-4113](common/arg.cpp#L4105-L4113)
- **Sets:** Port: `8014`, `use_jinja = true`

### `--vision-gemma-12b-default`
Use Gemma 3 12B QAT
- **Source:** [common/arg.cpp:4116-4124](common/arg.cpp#L4116-L4124)

---

## Environment Variables

Many parameters can be set via environment variables. Here's a comprehensive list:

| Environment Variable | Parameter | Default Value |
|---------------------|-----------|---------------|
| `LLAMA_ARG_HOST` | `--host` | `127.0.0.1` |
| `LLAMA_ARG_PORT` | `--port` | `8080` |
| `LLAMA_ARG_STATIC_PATH` | `--path` | `""` |
| `LLAMA_ARG_API_PREFIX` | `--api-prefix` | `""` |
| `LLAMA_API_KEY` | `--api-key` | none |
| `LLAMA_ARG_SSL_KEY_FILE` | `--ssl-key-file` | `""` |
| `LLAMA_ARG_SSL_CERT_FILE` | `--ssl-cert-file` | `""` |
| `LLAMA_CHAT_TEMPLATE_KWARGS` | `--chat-template-kwargs` | `{}` |
| `LLAMA_ARG_TIMEOUT` | `--timeout` | `600` |
| `LLAMA_ARG_THREADS_HTTP` | `--threads-http` | `-1` |
| `LLAMA_ARG_CACHE_REUSE` | `--cache-reuse` | `0` |
| `LLAMA_ARG_ENDPOINT_METRICS` | `--metrics` | disabled |
| `LLAMA_ARG_ENDPOINT_PROPS` | `--props` | disabled |
| `LLAMA_ARG_ENDPOINT_SLOTS` | `--slots` | enabled |
| `LLAMA_ARG_NO_ENDPOINT_SLOTS` | `--no-slots` | - |
| `LLAMA_ARG_NO_WEBUI` | `--no-webui` | - |
| `LLAMA_ARG_EMBEDDINGS` | `--embedding` | disabled |
| `LLAMA_ARG_RERANKING` | `--reranking` | disabled |
| `LLAMA_ARG_JINJA` | `--jinja` | disabled |
| `LLAMA_ARG_THINK` | `--reasoning-format` | `auto` |
| `LLAMA_ARG_THINK_BUDGET` | `--reasoning-budget` | `-1` |
| `LLAMA_ARG_CHAT_TEMPLATE` | `--chat-template` | model default |
| `LLAMA_ARG_CHAT_TEMPLATE_FILE` | `--chat-template-file` | model default |
| `LLAMA_ARG_NO_PREFILL_ASSISTANT` | `--no-prefill-assistant` | - |
| `LLAMA_ARG_ALIAS` | `--alias` | `""` |
| `LLAMA_ARG_MODEL` | `--model` | - |
| `LLAMA_ARG_MODEL_URL` | `--model-url` | - |
| `LLAMA_ARG_CTX_SIZE` | `--ctx-size` | `0` |
| `LLAMA_ARG_BATCH` | `--batch-size` | `2048` |
| `LLAMA_ARG_UBATCH` | `--ubatch-size` | `512` |
| `LLAMA_ARG_CTX_CHECKPOINTS` | `--ctx-checkpoints` | `8` |
| `LLAMA_ARG_CACHE_RAM` | `--cache-ram` | `8192` |
| `LLAMA_ARG_NO_CONTEXT_SHIFT` | `--no-context-shift` | - |
| `LLAMA_ARG_CONTEXT_SHIFT` | `--context-shift` | - |
| `LLAMA_ARG_THREADS` | `--threads` | auto |
| `LLAMA_ARG_FLASH_ATTN` | `--flash-attn` | `auto` |
| `LLAMA_ARG_N_PARALLEL` | `--parallel` | `1` |
| `LLAMA_ARG_CONT_BATCHING` | `--cont-batching` | disabled |
| `LLAMA_ARG_NO_CONT_BATCHING` | `--no-cont-batching` | - |
| `LLAMA_ARG_POOLING` | `--pooling` | model default |
| `LLAMA_ARG_CACHE_TYPE_K` | `--cache-type-k` | `f16` |
| `LLAMA_ARG_CACHE_TYPE_V` | `--cache-type-v` | `f16` |
| `LLAMA_ARG_NO_PERF` | `--no-perf` | false |
| `LLAMA_ARG_DRAFT_MAX` | `--draft-max` | `16` |
| `LLAMA_ARG_DRAFT_MIN` | `--draft-min` | `5` |
| `LLAMA_ARG_DRAFT_P_MIN` | `--draft-p-min` | `0.9` |
| `LLAMA_ARG_CTX_SIZE_DRAFT` | `--ctx-size-draft` | `0` |
| `LLAMA_ARG_N_GPU_LAYERS_DRAFT` | `--gpu-layers-draft` | `0` |
| `LLAMA_ARG_MODEL_DRAFT` | `--model-draft` | unused |
| `LLAMA_ARG_CACHE_TYPE_K_DRAFT` | `--cache-type-k-draft` | `f16` |
| `LLAMA_ARG_CACHE_TYPE_V_DRAFT` | `--cache-type-v-draft` | `f16` |

---

## Usage Examples

### Basic Server

Start a simple server with default settings:
```bash
llama-server -m models/my-model.gguf
```

### Server with API Key Authentication

```bash
llama-server -m models/my-model.gguf --api-key "your-secret-key"
```

### Server with SSL

```bash
llama-server -m models/my-model.gguf \
  --ssl-key-file /path/to/key.pem \
  --ssl-cert-file /path/to/cert.pem \
  --port 443
```

### Server with Custom Context and Performance Settings

```bash
llama-server -m models/my-model.gguf \
  -c 4096 \
  -t 8 \
  -ngl 32 \
  --cache-reuse 256 \
  --cont-batching
```

### Embedding Server

```bash
llama-server -m models/embedding-model.gguf \
  --embedding \
  --pooling mean \
  -c 512 \
  --port 8081
```

### Speculative Decoding Server

```bash
llama-server -m models/large-model.gguf \
  -md models/small-draft-model.gguf \
  --draft-max 16 \
  --draft-min 5 \
  -ngl 40 \
  -ngld 20
```

### Server with Monitoring Endpoints

```bash
llama-server -m models/my-model.gguf \
  --metrics \
  --slots \
  --props
```

---

## KV Cache Types

Supported KV cache data types:
- `f32`: 32-bit floating point
- `f16`: 16-bit floating point (default)
- `q8_0`: 8-bit quantized
- `q4_0`: 4-bit quantized
- `q4_1`: 4-bit quantized (variant 1)
- `iq4_nl`: 4-bit quantized (NL variant)
- `q5_0`: 5-bit quantized
- `q5_1`: 5-bit quantized (variant 1)

Lower precision types use less memory but may reduce quality slightly.

---

## Additional Resources

- **Main Source File:** [tools/server/server.cpp](tools/server/server.cpp)
- **Parameter Definitions:** [common/arg.cpp](common/arg.cpp)
- **Parameter Structures:** [common/common.h](common/common.h)
- **Server Utilities:** [tools/server/utils.hpp](tools/server/utils.hpp)

---

## Notes

1. **Parameter Priority:** Command-line arguments override environment variables
2. **Boolean Flags:** Most flags don't require a value (e.g., `--metrics`, `--no-webui`)
3. **Multiple Values:** Some parameters like `--api-key` can be specified multiple times
4. **Auto-download:** Some preset configurations automatically download models from the internet
5. **Security:** Advanced endpoints (`--props`, `--metrics`) are disabled by default for better security

---

*Generated from llama.cpp source code*
*Last updated: 2025-11-03*
