# QA bot with a local Qwen3-8B

A RAG chatbot that answers questions about the Markdown files you upload, with
everything running on your machine: no API keys, no cloud calls, and it keeps
working offline once set up.

```
03_qa_bot_with_local_qwen3.py (Gradio, http://127.0.0.1:7860)
   ├──► Ollama  (Docker, localhost:11434)
   │      ├─ qwen3-8b-rag        Qwen3-8B Q8_0, writes the answers
   │      └─ Qwen3-Embedding-0.6B turns text into vectors
   └──► Chroma  (Docker, localhost:8000)   stores and searches the vectors
```

| File                            | What it is                                                         |
| ------------------------------- | ------------------------------------------------------------------ |
| `03_qa_bot_with_local_qwen3.md` | The lesson notes: every step explained, including the why          |
| `03_qa_bot_with_local_qwen3.py` | The app                                                            |
| `compose.yaml`                  | The Ollama and Chroma services                                     |
| `Modelfile`                     | Builds `qwen3-8b-rag`: Qwen3-8B with a chat template whose thinking can be turned off |

## Requirements

- Docker Engine with the Compose plugin (`docker compose version`)
- The repo's Python environment (`make venv && make install` from the repo root)
- About 12 GB of free disk for the images and models
- Recommended: an NVIDIA GPU (the app was tested on 16 GB; it uses about 11 GB)

### GPU access from Docker

Docker containers can't use the GPU until the NVIDIA Container Toolkit is
installed. Check:

```bash
docker info | grep -i runtimes                 # must list "nvidia"
docker run --rm --gpus all ubuntu nvidia-smi   # must print your GPU
```

If `nvidia` is missing, follow "Give Docker access to the GPU" in the lesson
notes. **No NVIDIA GPU?** Delete the `deploy:` block from `compose.yaml`;
everything runs on the CPU, just more slowly.

## One-time setup

Run these from this folder, while online:

```bash
cd lessons/module02/03_qa_bot_with_local_qwen3

# 1. Start Ollama and Chroma in the background
docker compose up -d

# 2. Download the models from Hugging Face into the ollama volume
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q8_0               # ~8.7 GB
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0   # ~0.6 GB

# 3. Build qwen3-8b-rag from the Modelfile (reuses the downloaded weights, no extra disk)
docker compose cp Modelfile ollama:/Modelfile
docker compose exec ollama ollama create qwen3-8b-rag -f /Modelfile
```

Check the result:

```bash
docker compose exec ollama ollama list    # qwen3-8b-rag and both hf.co/... models
docker compose logs ollama | grep "inference compute"    # library=CUDA means the GPU was found
```

Step 3 matters: the template Ollama generates for the raw `hf.co/...` model
ignores "thinking off", so the model's reasoning would leak into every answer.

## Run the app

1. Make sure the services are up (after a reboot, run `docker compose up -d` again):

   ```bash
   docker compose ps
   ```

2. Start the app **from the repo root**, so `app.config` can be imported:

   ```bash
   PYTHONPATH=. .venv/bin/python lessons/module02/03_qa_bot_with_local_qwen3/03_qa_bot_with_local_qwen3.py
   ```

   In PyCharm, just run the file; it already puts the repo root on the path.

3. Open <http://127.0.0.1:7860>, upload one or more `.md` files, type a
   question, and press **Submit**.

The first question takes 15-35 s while Ollama loads the models into the GPU.
After that, answers take a few seconds. The models stay loaded for 30 minutes
after their last use. Questions the files don't cover get "I don't know".

Stop the app with `Ctrl+C`.

## Configuration

| Setting                     | Where                                  | Default                                      |
| --------------------------- | -------------------------------------- | -------------------------------------------- |
| LLM                         | `QWEN3_RAG_MODEL` in `app/config/envs.py`       | `qwen3-8b-rag`                               |
| Embedding model             | `QWEN3_EMBEDDING_MODEL` in `app/config/envs.py` | `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0`  |
| Ollama / Chroma address     | `OLLAMA_URL`, `CHROMA_HOST`, `CHROMA_PORT` in `app/config/envs.py` | `http://localhost:11434`, `localhost:8000` |
| Sampling, context, answer length | `get_llm()` in the script         | `temperature=0.7`, `num_ctx=8192`, `num_predict=512` |
| Chunk size, overlap         | `text_splitter()` in the script        | 1000 / 100 characters                        |
| Chunks retrieved per question | `retriever()` in the script          | `k=4`                                        |
| How long models stay loaded | `OLLAMA_KEEP_ALIVE` in `compose.yaml`  | `30m`                                        |
| Gradio port                 | `launch()` in the script               | `7860`                                       |

**Smaller GPU (8 GB)?** Use the `Q4_K_M` quantization (5 GB) instead of `Q8_0`:
change it in the pull command and on the `FROM` line of the `Modelfile`, then
redo setup steps 2 and 3.

## Offline use

Only the setup needs the internet. The models and the vectors live in Docker
volumes (`qabot_ollama`, `qabot_chroma`), the app is served on `127.0.0.1`
without a share link, and the Gradio and Chroma telemetry is turned off.

## Stop and clean up

```bash
docker compose stop      # stop the services, keep everything
docker compose down      # remove the containers, keep the models and the vectors
docker compose down -v   # also delete the volumes: setup must be redone
```

## Troubleshooting

| Symptom                                                   | Fix                                                                               |
| --------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `could not select device driver "nvidia"`                 | Install the NVIDIA Container Toolkit, or delete the `deploy:` block in `compose.yaml` |
| Slow answers; `docker compose exec ollama ollama ps` shows `CPU` | The container can't see the GPU: run `docker run --rm --gpus all ubuntu nvidia-smi` |
| `model "qwen3-8b-rag" not found`                          | Redo setup step 3 (also needed after `docker compose down -v`)                    |
| Answers contain reasoning text or a `</think>`            | `QWEN3_RAG_MODEL` points at the raw `hf.co/...` model; use `qwen3-8b-rag`        |
| `Could not connect to a Chroma server` / connection refused | The services are down: `docker compose up -d`, then `docker compose logs`      |
| `ModuleNotFoundError: No module named 'app'`              | Run from the repo root with `PYTHONPATH=.`                                        |
| `Cannot find empty port` / port 7860 in use               | Another app instance is still running; stop it                                    |
| Out of GPU memory                                         | Lower `num_ctx`, or use `Q4_K_M` (see Configuration)                              |
