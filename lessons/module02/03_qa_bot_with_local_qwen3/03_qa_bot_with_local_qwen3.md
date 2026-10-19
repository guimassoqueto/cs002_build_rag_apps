# Construct a Local QA Bot that Leverages LangChain and Qwen3-8B to Answer Questions from Markdown Files

**Estimated time needed: 60 minutes**

In this project, you will construct a question-answering (QA) bot. The bot uses LangChain and a large language model (LLM) to answer questions about the content of Markdown files you upload. To build a fully functional QA system, you'll combine a document loader, text splitters, an embedding model, a vector database, a retriever, and Gradio as the front-end interface.

Unlike a cloud-based bot, **everything in this project runs on your own machine**:

- the LLM is **Qwen3-8B**, downloaded once from Hugging Face and served by **Ollama**;
- the embedding model is **Qwen3-Embedding-0.6B**, served by the same Ollama instance;
- the vector database is **ChromaDB**, running as a server;
- Ollama and ChromaDB both run in **Docker** containers, and your Python script talks to them over `localhost`.

There are no API keys, no accounts, and no per-token costs. Once the models are downloaded, the bot keeps working with the network cable unplugged.

Imagine you're tasked with creating an intelligent assistant that can quickly and accurately answer questions about a team's documentation: READMEs, runbooks, design notes, or course notes, all written in Markdown. Searching through these by hand is slow, and sending them to a third-party API may not be allowed. A local QA bot solves both problems.

## Learning objectives

By the end of this project, you will be able to:

- Run an open-weight LLM and an embedding model locally with Ollama in Docker, with GPU acceleration
- Explain why a quantized model (GGUF Q8_0) fits on a consumer GPU when the full-precision one does not
- Combine a document loader, Markdown-aware text splitters, an embedding model, and a vector database to construct a fully functional QA bot
- Build a retrieval-augmented generation (RAG) chain with the LangChain Expression Language (LCEL)
- Serve the bot through a Gradio interface that works fully offline

## Architecture

```
                 your machine
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  qabot.py (Python + Gradio, http://127.0.0.1:7860)           │
│     │  upload .md files, ask a question                      │
│     │                                                        │
│     ├──► Ollama container  (localhost:11434)                 │
│     │      ├─ Qwen3-8B Q8_0            → writes the answer   │
│     │      └─ Qwen3-Embedding-0.6B     → turns text into     │
│     │                                    vectors             │
│     │                                                        │
│     └──► Chroma container  (localhost:8000)                  │
│            └─ stores the chunk vectors, finds the closest    │
│               ones to the question                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

# Set up

## Choosing the model: why Qwen3-8B Q8_0

[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) is an 8-billion-parameter open-weight model (Apache 2.0). The memory a model needs is roughly *parameters × bytes per parameter*:

| Version                                    | Bytes / parameter | Weights size | Fits in a 16 GB GPU?       |
| ------------------------------------------ | ----------------- | ------------ | -------------------------- |
| `Qwen/Qwen3-8B` (bf16, full precision)     | 2                 | ~16.4 GB     | No - nothing left for work |
| `Qwen/Qwen3-8B-GGUF` **Q8_0**              | ~1                | **8.71 GB**  | **Yes** - near-lossless    |
| `Qwen/Qwen3-8B-GGUF` Q4_K_M                | ~0.5              | 5.03 GB      | Yes - small quality loss   |

[`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF) is the official quantized release from the Qwen team: the same model, stored with 8 bits per weight instead of 16. Besides the weights, the GPU also holds the *KV cache* (the model's working memory for the current prompt, about 1.2 GB for an 8K-token context) and the embedding model (about 0.6 GB). The total, about 10.5 GB, leaves comfortable headroom on a 16 GB card.

If you have a smaller GPU (8 GB), use `Q4_K_M` instead of `Q8_0` everywhere in this lab. Without a GPU at all, everything still works on the CPU, just more slowly.

> **Thinking mode.** Qwen3 is a *hybrid reasoning* model: by default it writes its reasoning inside `<think>...</think>` before answering. That is useful for math and multi-step problems, but for answering from retrieved context it mostly adds latency (about 25 s instead of 5 s per answer in testing), so this lab turns it off. Doing that reliably takes one extra setup step, described in [Create a local model with switchable thinking](#create-a-local-model-with-switchable-thinking).

## Prerequisites

- **Docker Engine** with the **Compose** plugin (`docker compose version` should work)
- **Python 3.11+** (this lab was tested with Python 3.14)
- About **12 GB of free disk** for the images and models
- *Optional, recommended:* an **NVIDIA GPU** with a recent driver (`nvidia-smi` should work on the host)

## Give Docker access to the GPU (NVIDIA Container Toolkit)

Skip this section if you have no NVIDIA GPU; Ollama will use the CPU.

Having the NVIDIA driver on the host is not enough for containers to use the GPU. GPU access involves three layers:

```
container image  →  CUDA user-space libraries (libcudart, cuBLAS, ...)   shipped inside the image
      ↕   ???
host             →  driver library libcuda.so + device files /dev/nvidia* from the nvidia-driver package
kernel           →  nvidia.ko kernel module
```

A container is an isolated filesystem with an isolated device list. By default, Docker gives it neither the `/dev/nvidia*` device files nor the host's `libcuda.so`, and an image cannot ship its own `libcuda.so` because that library must match the host's kernel driver exactly. Without help, the CUDA code inside the container finds no GPU and silently falls back to the CPU.

The **NVIDIA Container Toolkit** fills the `???` gap. It registers an `nvidia` runtime with Docker; when a container requests a GPU (`--gpus all`, or the `deploy.resources.reservations.devices` block in Compose), the toolkit mounts the device files and the host's driver libraries into that container. Nothing is installed inside the image, and the host driver stays untouched.

Check whether Docker already knows the runtime:

```shell
docker info | grep -i runtimes    # "nvidia" should be listed
```

If it isn't, install the toolkit (Debian/Ubuntu shown; see the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for other distributions):

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker   # adds the "nvidia" runtime to /etc/docker/daemon.json
sudo systemctl restart docker
```

Verify that a container can see the GPU. This should print the same table as `nvidia-smi` on the host:

```shell
docker run --rm --gpus all ubuntu nvidia-smi
```

## Start the local services

Create a file named `compose.yaml` with the two services the bot needs:

```yaml
name: qabot

services:
  ollama:
    image: ollama/ollama:0.40.6
    ports:
      - "127.0.0.1:11434:11434"
    volumes:
      - ollama:/root/.ollama
    environment:
      OLLAMA_KEEP_ALIVE: 30m
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  chroma:
    image: chromadb/chroma:1.5.9
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - chroma:/data
    environment:
      ANONYMIZED_TELEMETRY: "False"

volumes:
  ollama:
  chroma:
```

What each part does:

- **`ollama`** serves both models over an HTTP API on port 11434. Downloaded models live in the `ollama` volume, so they survive container restarts and are downloaded only once. `OLLAMA_KEEP_ALIVE: 30m` keeps a model loaded in GPU memory for 30 minutes after its last request, so you don't pay the load time on every question.
- **`deploy.resources.reservations.devices`** is the Compose equivalent of `docker run --gpus all`. **Without a GPU, delete this block**, or the container will fail to start.
- **`chroma`** is the vector database server on port 8000. Its data lives in the `chroma` volume. `ANONYMIZED_TELEMETRY: "False"` stops it from trying to reach the internet.
- **`127.0.0.1:`** in each port mapping exposes the services to your machine only, not to the rest of the network.
- The image versions are **pinned**, so the lab keeps working the same way when new releases come out. The `chromadb/chroma` version matches the `chromadb` Python client installed below.

Start both services in the background and check that they are running:

```shell
docker compose up -d
docker compose ps
curl http://localhost:11434/api/version      # {"version":"0.40.6"}
curl http://localhost:8000/api/v2/heartbeat  # {"nanosecond heartbeat": ...}
```

Confirm that Ollama found the GPU. You should see a line with `library=CUDA` and your GPU's name:

```shell
docker compose logs ollama | grep "inference compute"
```

## Download the models (once)

Ollama can pull GGUF models straight from Hugging Face with the `hf.co/<user>/<repo>:<quantization>` syntax. Run these once, while you are online:

```shell
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q8_0
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
docker compose exec ollama ollama list
```

- `Qwen3-8B-GGUF:Q8_0` (8.7 GB) is the LLM that writes the answers.
- [`Qwen3-Embedding-0.6B-GGUF:Q8_0`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) (639 MB) is the embedding model. It turns text into 1024-dimensional vectors and supports inputs of up to 32K tokens.

## Create a local model with switchable thinking

When Ollama pulls a GGUF from Hugging Face, it converts the model's Jinja chat template into its own Go template format. For Qwen3-8B, the converted template ends by **always** opening a `<think>` block for the model's reply:

```
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
<think>
{{ end }}
```

As a result, the API's `think: false` option (which LangChain sends as `reasoning=False`) is ignored: the model still reasons, and the reasoning text leaks into the answer, followed by a stray `</think>`. Qwen's original template handles this differently: when thinking is disabled, it pre-fills an **empty** `<think>\n\n</think>` block, so the model goes straight to the answer.

You can fix this without touching the weights. A **Modelfile** is Ollama's recipe for a model: `FROM` names the weights, and `TEMPLATE` overrides the chat template. Create a file named `Modelfile` next to `compose.yaml`. It is the converted template with only the last block changed: an empty `<think>` block when thinking is turned off, and the usual open `<think>` otherwise.

```
FROM hf.co/Qwen/Qwen3-8B-GGUF:Q8_0

TEMPLATE """{{- $lastUserIdx := -1 -}}
{{- range $idx, $msg := .Messages -}}
{{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
{{- end }}
{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}{{ .System }}

{{ end }}
{{- if .Tools }}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $.IsThinkSet (and .Thinking (or $last (gt $i $lastUserIdx)))) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{ if .Content }}{{ .Content }}{{ end }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ if and $.IsThinkSet (not $.Think) }}<think>

</think>

{{ else }}<think>
{{ end }}{{ end }}
{{- end }}"""
```

`$.IsThinkSet` is true when the request sets `think` at all, and `$.Think` holds its value, so the empty block is used only when a request explicitly asks for `think: false`.

Copy the Modelfile into the container and create a model named `qwen3-8b-rag` from it:

```shell
docker compose cp Modelfile ollama:/Modelfile
docker compose exec ollama ollama create qwen3-8b-rag -f /Modelfile
docker compose exec ollama ollama list
```

`qwen3-8b-rag` reuses the 8.7 GB weights file that is already downloaded; only the small template is new, so it costs no extra disk space. It also inherits the sampling defaults published in the Hugging Face repo, such as `presence_penalty 1.5`, which Qwen recommends for quantized models.

Give the LLM a quick try, with thinking off and then on:

```shell
docker compose exec ollama ollama run qwen3-8b-rag --think=false "In one sentence, what is RAG?"
docker compose exec ollama ollama run qwen3-8b-rag --think=true "What is 17*23?"
docker compose exec ollama ollama ps    # PROCESSOR should read "100% GPU"
```

With `--think=false` you get only the answer. With `--think=true` you first see the reasoning, then the answer.

## Set up the Python environment

Create and activate a virtual environment:

```shell
python3 -m venv .venv
source .venv/bin/activate
```

Install the libraries the bot needs:

- [**`gradio`**](https://www.gradio.app/) builds the web interface.
- [**`langchain-core`**](https://python.langchain.com/) provides documents, prompts, and the LCEL building blocks.
- [**`langchain-text-splitters`**](https://python.langchain.com/docs/concepts/text_splitters/) splits Markdown into chunks.
- [**`langchain-ollama`**](https://python.langchain.com/docs/integrations/providers/ollama/) connects LangChain to the local Ollama server, for both chat and embeddings.
- [**`langchain-chroma`**](https://python.langchain.com/docs/integrations/vectorstores/chroma/) connects LangChain to Chroma, and installs the `chromadb` client.

```shell
python -m pip install \
  gradio==6.30.0 \
  langchain-core==1.6.12 \
  langchain-text-splitters==1.1.3 \
  langchain-ollama==1.1.0 \
  langchain-chroma==1.1.0 \
  chromadb==1.5.9
```

The environment is now ready for the application.

# Construct the QA bot

It's time to construct the QA bot! Create a new Python file named `qabot.py`. You will fill it in over the following sections.

## Import the necessary libraries

Inside `qabot.py`, add the imports below. They cover loading files, splitting text, connecting to Ollama and Chroma, building the question-answering chain, and creating the Gradio interface.

```python
from pathlib import Path

import chromadb
import gradio as gr
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
```

Then add the settings for the local services and models in one place, so they are easy to change. `LLM_MODEL` is the model you created from the Modelfile:

```python
OLLAMA_URL = "http://localhost:11434"
CHROMA_HOST, CHROMA_PORT = "localhost", 8000
LLM_MODEL = "qwen3-8b-rag"
EMBEDDING_MODEL = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
```

## Initialize the LLM

Initialize the LLM by creating an instance of `ChatOllama`, the LangChain chat model class for Ollama. It sends each request to the local Ollama server, which runs Qwen3-8B on your GPU. Any model you have pulled into Ollama can be used here by changing `LLM_MODEL`.

```python
## LLM
def get_llm():
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        reasoning=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        num_ctx=8192,
        num_predict=512,
    )
    return llm
```

The parameters:

- **`reasoning=False`** turns off Qwen3's thinking mode, so the model answers directly. It works because `qwen3-8b-rag` has the fixed template; with the raw `hf.co/...` model the reasoning would leak into the answer.
- **`temperature=0.7`, `top_p=0.8`, `top_k=20`** are the Qwen team's recommended sampling settings for non-thinking mode. Qwen advises against greedy decoding (`temperature=0`), which can make Qwen3 repeat itself.
- **`num_ctx=8192`** is the context window in tokens: the instructions, the retrieved chunks, and the question all have to fit in it. Qwen3-8B supports up to 32K natively; a larger window needs more GPU memory for the KV cache.
- **`num_predict=512`** is the maximum number of tokens in the answer.

## Define the Markdown document loader

Next, define the document loader. Markdown files are plain text, so no special parser is needed: read each file and wrap its content in a LangChain `Document`. The file name goes into the document's `metadata`, so every chunk created later still knows which file it came from.

```python
## Document loader
def document_loader(files):
    loaded_documents = [
        Document(page_content=Path(file).read_text(encoding="utf-8"), metadata={"source": Path(file).name})
        for file in files
    ]
    return loaded_documents
```

`files` is a list of file paths, because the Gradio interface below lets you upload several Markdown files at once.

## Define the text splitter

The loader returns each file as a single document, so you must split the text into chunks small enough to embed and retrieve precisely. Markdown already has structure, so the splitter works in two stages:

1. **`MarkdownHeaderTextSplitter`** cuts each file at its `#`, `##`, and `###` headings, so a chunk never mixes two unrelated sections. Each heading is recorded in the chunk's metadata (`h1`, `h2`, `h3`), and `strip_headers=False` keeps the heading in the text too, which gives the embedding model useful context.
2. **`RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)`** further splits any section longer than 1000 characters, preferring Markdown boundaries (headings, code blocks, paragraphs) over cutting mid-sentence. The 100-character overlap preserves context across chunk boundaries.

```python
## Text splitter
def text_splitter(data):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = []
    for document in data:
        for section in header_splitter.split_text(document.page_content):
            section.metadata |= document.metadata
            sections.append(section)

    chunk_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = chunk_splitter.split_documents(sections)
    return chunks
```

`section.metadata |= document.metadata` copies the `source` file name onto each section, because `split_text()` only receives the text.

## Define the vector store

Now that you can load the Markdown files and split them into chunks, you need a way to embed and store those chunks in a vector database. The function below connects to the Chroma server running in Docker, embeds the chunks with a yet-to-be-defined embedding model, and stores them in a collection named `qabot`:

```python
## Vector db
def vector_database(chunks):
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    vectordb = Chroma(
        collection_name="qabot",
        embedding_function=local_embedding(),
        client=client,
    )
    vectordb.reset_collection()
    vectordb.add_documents(chunks)
    return vectordb
```

- **`chromadb.HttpClient`** talks to the Chroma container. `anonymized_telemetry=False` keeps the client from reaching out to the internet.
- **`reset_collection()`** empties the collection before adding the new chunks. The Chroma server is persistent, so without it, chunks from earlier uploads would pile up and leak into answers about other files.

## Define the embedding model

The `vector_database()` function above relies on a `local_embedding()` function that returns an embedding model, which converts chunks of text into vectors. `OllamaEmbeddings` sends the text to the same local Ollama server, which runs Qwen3-Embedding-0.6B:

```python
## Embedding model
def local_embedding():
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )
    return embedding
```

It does not matter to Python that `local_embedding()` is defined after `vector_database()`; the function is only looked up when `vector_database()` runs.

Chroma uses the embedding model twice: once to embed every chunk when it is added, and once to embed the question at query time. Both must use the **same** model, or the vectors would not be comparable.

## Define the retriever

With the vector store defined, you can define a retriever that fetches the chunks most similar to a question. `as_retriever()` turns the vector store into a retriever that runs a similarity search; `k=4` returns the four closest chunks.

```python
## Retriever
def retriever(files):
    splits = document_loader(files)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever
```

## Define a question-answering chain

Finally, it's time to define the question-answering chain. It performs retrieval-augmented generation (RAG): it retrieves the relevant chunks, "stuffs" them into the prompt as context, and asks the LLM to answer the question from that context.

First, the prompt. The system message holds the retrieved context and tells the model to stay within it:

```python
## Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Answer the question using only the context below. "
            "If the answer is not in the context, say that you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
```

Then the chain itself, built with the LangChain Expression Language (LCEL), where `|` passes each step's output to the next step:

```python
## QA Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retriever_qa(files, query):
    llm = get_llm()
    retriever_obj = retriever(files)
    qa_chain = (
        {"context": retriever_obj | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    response = qa_chain.invoke(query)
    return response
```

Read the chain from left to right:

1. The dictionary runs two branches on the query: `retriever_obj | format_docs` fetches the four closest chunks and joins them into one string (`context`), while `RunnablePassthrough()` passes the query through unchanged (`question`).
2. `prompt` fills `{context}` and `{question}` into the chat messages.
3. `llm` sends the messages to Qwen3-8B.
4. `StrOutputParser()` extracts the answer text from the model's reply.

Let's recap how the elements of the bot are linked. The chain needs an LLM (`get_llm()`) and a retriever (`retriever()`). The retriever is based on the vector store (`vector_database()`), which in turn needs an embedding model (`local_embedding()`) and chunks produced by the text splitter (`text_splitter()`). The text splitter needs raw documents, which the loader (`document_loader()`) reads from the Markdown files. This defines the core functionality of your QA bot!

## Set up the Gradio interface

With the core functionality in place, the last piece is the Gradio interface. It should include:

- A file upload that accepts one or more Markdown files (the `File` class in Gradio)
- An input textbox where the question is asked (the `Textbox` class in Gradio)
- An output textbox where the answer appears (the `Textbox` class in Gradio)

Add the following code to `qabot.py`:

```python
# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    analytics_enabled=False,
    inputs=[
        gr.File(label="Upload Markdown files", file_count="multiple", file_types=[".md"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Output"),
    title="Local RAG Chatbot",
    description="Upload Markdown documents and ask any question. The chatbot answers using only the provided documents, with a model running on your machine.",
)
```

- **`file_count="multiple"`** with **`type="filepath"`** hands `retriever_qa()` a list of paths to the uploaded files.
- **`flagging_mode="never"`** hides the Flag button.
- **`analytics_enabled=False`** stops Gradio from sending usage analytics, so the app makes no outgoing requests.

## Add code to launch the application

Finally, add one more line to `qabot.py` to launch the application on port 7860:

```python
# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port=7860)
```

`server_name="127.0.0.1"` serves the app to your machine only. There is no `share=True`: a share link is tunneled through Gradio's servers, which needs the internet and makes your app reachable from outside.

After adding this line, save `qabot.py`.

## Verify `qabot.py`

Your `qabot.py` should now look like the following:

```python
from pathlib import Path

import chromadb
import gradio as gr
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

OLLAMA_URL = "http://localhost:11434"
CHROMA_HOST, CHROMA_PORT = "localhost", 8000
LLM_MODEL = "qwen3-8b-rag"
EMBEDDING_MODEL = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"


## LLM
def get_llm():
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        reasoning=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        num_ctx=8192,
        num_predict=512,
    )
    return llm


## Document loader
def document_loader(files):
    loaded_documents = [
        Document(page_content=Path(file).read_text(encoding="utf-8"), metadata={"source": Path(file).name})
        for file in files
    ]
    return loaded_documents


## Text splitter
def text_splitter(data):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = []
    for document in data:
        for section in header_splitter.split_text(document.page_content):
            section.metadata |= document.metadata
            sections.append(section)

    chunk_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = chunk_splitter.split_documents(sections)
    return chunks


## Vector db
def vector_database(chunks):
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    vectordb = Chroma(
        collection_name="qabot",
        embedding_function=local_embedding(),
        client=client,
    )
    vectordb.reset_collection()
    vectordb.add_documents(chunks)
    return vectordb


## Embedding model
def local_embedding():
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )
    return embedding


## Retriever
def retriever(files):
    splits = document_loader(files)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever


## Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Answer the question using only the context below. "
            "If the answer is not in the context, say that you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


## QA Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retriever_qa(files, query):
    llm = get_llm()
    retriever_obj = retriever(files)
    qa_chain = (
        {"context": retriever_obj | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    response = qa_chain.invoke(query)
    return response


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    analytics_enabled=False,
    inputs=[
        gr.File(label="Upload Markdown files", file_count="multiple", file_types=[".md"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Output"),
    title="Local RAG Chatbot",
    description="Upload Markdown documents and ask any question. The chatbot answers using only the provided documents, with a model running on your machine.",
)

# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port=7860)
```

# Serve the application

Make sure the Docker services are running (`docker compose ps`), then start the bot from the terminal where your virtual environment is active:

```shell
python qabot.py
```

If you open a new terminal, activate the virtual environment first:

```shell
source .venv/bin/activate
```

# Launch the application

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser. Upload one or more Markdown files, type a question about their content, and press **Submit**.

The first question after starting the services takes longer (about 15-35 s), because Ollama has to load both models into GPU memory. Later questions reuse the loaded models until they have been idle for 30 minutes (`OLLAMA_KEEP_ALIVE`); on a 16 GB RTX-class GPU, a typical answer then takes 3-6 seconds.

You can test the bot with any Markdown notes you have, for example the course notes from the earlier Gradio lessons. Ask something the notes cover, then something they don't (such as "What is the capital of France?"): the bot should answer the first and say that it doesn't know the second.

The current setup re-reads, re-splits, and re-embeds the uploaded files on every question, as in a simple prototype. For a few Markdown files this takes under a second; for a large documentation set, embedding once and querying many times would be the next improvement.

When you finish experimenting, press `Ctrl+C` in the terminal to stop the bot.

# Run fully offline

Only the setup needs the internet: pulling the two Docker images, the two models, and the Python packages. After that, nothing in the stack contacts the outside world:

| Component | Why it works offline                                                                    |
| --------- | --------------------------------------------------------------------------------------- |
| Ollama    | Models are stored in the `ollama` Docker volume; inference runs on your GPU/CPU         |
| Chroma    | Data is stored in the `chroma` Docker volume; telemetry is disabled on server and client |
| Gradio    | Served on `127.0.0.1`, no `share=True`, analytics disabled                              |

To check it, disconnect from the network, restart everything, and ask a question:

```shell
docker compose restart
python qabot.py
```

# Stop and clean up

```shell
docker compose stop      # stop the services, keep everything
docker compose down      # remove the containers, keep the models and the vectors (volumes)
docker compose down -v   # also delete the volumes: the models must be downloaded again
```

# Troubleshooting

| Symptom                                                     | Likely cause and fix                                                                                                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `could not select device driver "nvidia"` on `compose up`   | The NVIDIA Container Toolkit is missing or Docker was not restarted. Install it (see above), or delete the `deploy` block to run on the CPU.            |
| Answers are slow; `ollama ps` shows `CPU` instead of `GPU`  | The container can't see the GPU. Run `docker run --rm --gpus all ubuntu nvidia-smi` to check the toolkit.                                              |
| `model "hf.co/..." not found`                               | The model was not pulled. Run the `ollama pull` commands, and check the exact name with `ollama list`.                                                 |
| `Could not connect to a Chroma server`                      | The `chroma` service is not running. Check `docker compose ps` and `docker compose logs chroma`.                                                       |
| The answer contains the model's reasoning or a `</think>`   | `LLM_MODEL` points at the raw `hf.co/...` model, or `reasoning=False` is missing. Use `qwen3-8b-rag` from the Modelfile step.                          |
| `model "qwen3-8b-rag" not found`                            | The `ollama create` step was skipped, or the volume was deleted with `down -v`. Rerun the Modelfile step.                                             |
| Out of GPU memory                                           | Use the `Q4_K_M` quantization, or lower `num_ctx`.                                                                                                     |

# What changed from the original IBM watsonx lab

| Piece           | Original lab                                       | This lab                                                       |
| --------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| LLM             | `mistralai/mistral-medium-2505` on IBM watsonx.ai   | `Qwen3-8B` Q8_0, local, served by Ollama in Docker (`qwen3-8b-rag`) |
| Embeddings      | `ibm/granite-embedding-278m-multilingual` on watsonx | `Qwen3-Embedding-0.6B` Q8_0, local, served by Ollama            |
| Vector database | Chroma, in memory inside the Python process        | Chroma server in Docker, persistent volume                     |
| Input documents | One PDF (`PyPDFLoader`)                            | One or more Markdown files (`Document` + Markdown splitters)    |
| Text splitting  | `RecursiveCharacterTextSplitter`                   | `MarkdownHeaderTextSplitter`, then a Markdown-aware recursive splitter |
| QA chain        | `RetrievalQA` (legacy chain)                       | LCEL chain: retriever → prompt → LLM → output parser          |
| Credentials     | watsonx project ID                                 | None                                                           |
| Network         | Required for every question; public `share=True` link | Required only for setup; served on `127.0.0.1`                 |

## Author(s)

[Kang Wang](https://author.skills.network/instructors/kang_wang)

[Wojciech "Victor" Fulmyk](https://www.linkedin.com/in/wfulmyk)

[Kunal Makwana](https://author.skills.network/instructors/kunal_makwana)

[Ricky Shi](https://author.skills.network/instructors/ricky_shi)

## Contributor(s)

[Hailey Quach](https://author.skills.network/instructors/hailey_quach)

*Adapted to run fully locally with Qwen3-8B, Ollama, and Chroma, and to answer questions from Markdown files.*
