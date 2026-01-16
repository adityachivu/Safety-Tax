# Persistent vLLM inference server on Modal
# 
# This script deploys a persistent OpenAI-compatible vLLM server.
# The server will stay running until explicitly stopped.
#
# To deploy:
#   modal deploy modal/vllm_inference.py
#
# The server will be available at the URL shown after deployment.
#
# To use with the client script:
#   python modal-examples/06_gpu_and_ml/llm-serving/openai_compatible/client.py \
#     --app-name vllm-inference-server \
#     --function-name serve
#
# Or use any OpenAI-compatible client by pointing it to:
#   https://your-workspace--vllm-inference-server-serve.modal.run/v1

import modal

# Set up the container image
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

# Model configuration
# MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507-FP8"
# MODEL_REVISION = "953532f942706930ec4bb870569932ef63038fdf"  # avoid nasty surprises when repos update!
# MODEL_NAME = "ArliAI/gpt-oss-20b-Derestricted"
# MODEL_REVISION = "main"  # avoid nasty surprises when repos update!
MODEL_NAME = "openai/gpt-oss-20b"
MODEL_REVISION = "main"  # avoid nasty surprises when repos update!

# Cache volumes for model weights and vLLM compilation artifacts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Configuration
FAST_BOOT = True  # Set to False for better performance if server stays warm

app = modal.App("vllm-inference-server-gpt-oss-20b")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Start the vLLM server. This function runs persistently once deployed."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print("Starting vLLM server with command:")
    print(" ".join(cmd))

    # Start the vLLM server process
    # The @web_server decorator keeps the container alive and handles routing
    subprocess.Popen(" ".join(cmd), shell=True)
