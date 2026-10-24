# Icebreaker bot with a local Qwen3

A RAG bot that reads a LinkedIn profile, gives you three interesting facts about
the person, and answers your questions about them. LlamaIndex does the
splitting, indexing and retrieval; Qwen3-8B and Qwen3-Embedding-0.6B run on
your machine in Ollama. No API keys, no cloud LLM.

It is the IBM lab "Build an AI Icebreaker Bot with LlamaIndex & IBM Granite"
with watsonx.ai replaced by a local Qwen3. The lab, every file explained step by
step, is [04_icebreaker_bot_with_local_qwen3.md](04_icebreaker_bot_with_local_qwen3.md).

```
main.py (CLI) / app.py (Gradio, http://127.0.0.1:5000)
   ├──► mock profile JSON (downloaded on each run)
   ├──► LlamaIndex in-memory vector index
   └──► Ollama (Docker, localhost:11434)
          ├─ qwen3-8b-rag          Qwen3-8B Q8_0, writes the facts and answers
          └─ Qwen3-Embedding-0.6B  turns text into vectors
```

This is a self-contained project: it has its own `.venv` and does not use the
repo's `.venv` or `app/config`. Run everything from this folder.

## Requirements

- Python 3.14
- Docker Engine with the Compose plugin (`docker compose version`)
- About 10 GB of free disk for the Ollama image and the models
- Recommended: an NVIDIA GPU with the NVIDIA Container Toolkit (it uses about
  11 GB). No NVIDIA GPU? Delete the `deploy:` block from `compose.yaml`;
  everything runs on the CPU, just more slowly.

## Setup

```bash
cd lessons/module03/04_icebreaker_bot_with_local_qwen3
make venv       # create .venv
make install    # install the project and its dependencies into .venv
make env        # copy .env.sample to .env (won't overwrite an existing one)
make models     # start Ollama, download the models (~9 GB), build qwen3-8b-rag
```

`make models` is a one-time step that needs the internet. The models live in
the `icebreaker_ollama` Docker volume.

## Run

```bash
make up         # start Ollama (after a reboot)
make cli        # command-line bot on the mock profile; press Enter at the URL prompt
make app        # web app on http://127.0.0.1:5000
```

In the web app, leave **Use Mock Data** checked, click **Process Profile**, then
ask questions on the **Chat** tab. The first request takes 15-35 s while Ollama
loads the models; after that, answers take a few seconds. Stop with `Ctrl+C`.

## Commands

| Command             | Does                                                              |
| ------------------- | ----------------------------------------------------------------- |
| `make venv`         | Create `.venv`                                                    |
| `make install`      | `pip install -e ".[dev]"`, then make `.venv` writable only by you |
| `make env`          | Create `.env` from `.env.sample`                                  |
| `make up` / `down`  | Start / remove the Ollama container (the models are kept)         |
| `make models`       | Start Ollama, pull both models, build `qwen3-8b-rag`              |
| `make check-config` | Print the main settings from `config.py`                          |
| `make cli`          | `python main.py --mock`                                           |
| `make app`          | `python app.py`                                                   |
| `make lint`         | Lint and format with ruff, applying fixes                         |
| `make lint-check`   | Lint and format checks only                                       |

## Configuration

| Setting                          | Where                                           | Default                                     |
| -------------------------------- | ----------------------------------------------- | ------------------------------------------- |
| LLM, embedding model             | `LLM_MODEL_ID`, `EMBEDDING_MODEL_ID` in `config.py` | `qwen3-8b-rag`, `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` |
| Ollama address, timeout          | `OLLAMA_URL`, `REQUEST_TIMEOUT` in `config.py`  | `http://localhost:11434`, 300 s             |
| Context window (`num_ctx`)       | `CONTEXT_WINDOW` in `config.py`                 | 8192                                        |
| Chunk size, chunks retrieved     | `CHUNK_SIZE`, `SIMILARITY_TOP_K` in `config.py` | 500 tokens, 5                               |
| Prompts                          | `INITIAL_FACTS_TEMPLATE`, `USER_QUESTION_TEMPLATE` in `config.py` |                           |
| Proxycurl API key                | `PROXYCURL_API_KEY` in `.env`                   | empty (Proxycurl was discontinued; use the mock data) |
| Models in the web app's dropdown | `available_models` in `app.py`                  | `qwen3-8b-rag`                              |
| How long models stay loaded      | `OLLAMA_KEEP_ALIVE` in `compose.yaml`           | `30m`                                       |

**Smaller GPU (8 GB)?** Use the `Q4_K_M` quantization (5 GB): change it in the
Makefile's `LLM_BASE` and on the `FROM` line of the `Modelfile`, then run
`make models` again.

## Stop and clean up

```bash
docker compose stop      # stop Ollama, keep everything
make down                # remove the container, keep the models
docker compose down -v   # also delete the models: `make models` must be redone
```

## Troubleshooting

| Symptom                                                   | Fix                                                                         |
| --------------------------------------------------------- | --------------------------------------------------------------------------- |
| `Connection refused` / `Failed to connect to Ollama`      | `make up`                                                                   |
| `make up`: `port is already allocated` (11434)            | Another Ollama is running, e.g. module02's: `docker compose down` in its folder |
| `model "qwen3-8b-rag" not found`                          | `make models` (also needed after `docker compose down -v`)                  |
| `Created 0 nodes`, and `NLTK will not read data` warnings | `chmod -R go-w .venv` (`make install` does it)                              |
| Answers contain reasoning text or a `</think>`            | `LLM_MODEL_ID` points at the raw `hf.co/...` model; use `qwen3-8b-rag`      |
| `ModuleNotFoundError: No module named 'config'`           | Run from this folder                                                        |
| `Failed to retrieve profile data`                         | The mock profile could not be downloaded: check your internet connection    |
| `could not select device driver "nvidia"`                 | Install the NVIDIA Container Toolkit, or delete the `deploy:` block in `compose.yaml` |
| Out of GPU memory                                         | Lower `CONTEXT_WINDOW`, or use `Q4_K_M` (see Configuration)                 |
