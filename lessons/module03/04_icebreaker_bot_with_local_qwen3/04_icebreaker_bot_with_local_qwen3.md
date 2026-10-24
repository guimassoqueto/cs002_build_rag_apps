# Build an AI Icebreaker Bot with LlamaIndex & a Local Qwen3

**Estimated time needed:** **_45_** minutes (plus a one-time download of about 9 GB of models)

Imagine you're heading to a big networking event, surrounded by potential employers and industry leaders. You want to make a great first impression, but you're struggling to come up with more than the usual, "What do you do?"

Now, picture having an AI-powered tool that does the research for you. You input a name, and within seconds, the bot, powered by **LlamaIndex** and a **Qwen3** model running **on your own machine**, reads a LinkedIn profile and generates personalized icebreakers based on someone's career highlights, interests, and even fun facts. Instead of generic questions, you start with something unique and meaningful.

The AI icebreaker bot uses **natural language processing (NLP)** to create tailored conversation starters that help you stand out. By the end of this project, you'll have built a tool that helps make introductions smoother, more personal, and more memorable for networking, job interviews, or any social setting. No cloud LLM, no API key: the language model and the embedding model both run locally in Ollama.

* * *

After completing this lab, you will be able to:

-   Understand how to use **LlamaIndex** with **local models served by Ollama** for personalized information retrieval.

-   Learn how to search and extract relevant data from **LinkedIn** for icebreaker generation.

-   Develop skills in using **AI and natural language processing (NLP)** to generate customized conversation starters based on a person's online presence.

-   Gain practical experience in applying **AI-powered tools** to automate social interactions, making introductions more engaging and memorable.

* * *

### **Disclaimer:** As of February 2025, Proxycurl has been discontinued, so the online approach requiring a valid Proxycurl API key is no longer functional. However, the offline option using mock data remains fully functional. The remainder of the lab still effectively demonstrates how to build a RAG application with LlamaIndex and a local Qwen3, including the option to integrate an API. Please use the mock data to complete the lab. We have retained the code sections demonstrating API usage, should you wish to substitute Proxycurl with another service for your own development purposes.

* * *

## What changed from the original lab

This is the IBM Skills Network lab "Build an AI Icebreaker Bot with LlamaIndex & IBM Granite", rewritten to run entirely on your machine. The application, the RAG pipeline, the files and the functions are the same; only the model layer and a few small bugs changed.

| Original lab                                                   | This version                                                                                   |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| LLM `ibm/granite-4-h-small` or `meta-llama/llama-3-3-70b-instruct` on IBM watsonx.ai | `qwen3-8b-rag` (Qwen3-8B Q8_0) served locally by Ollama                     |
| Embeddings `ibm/granite-embedding-278m-multilingual` on watsonx.ai | `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` served locally by Ollama                     |
| `llama-index-llms-ibm`, `llama-index-embeddings-ibm`, `ibm-watsonx-ai` | `llama-index-llms-ollama`, `llama-index-embeddings-ollama`                               |
| `create_watsonx_embedding()`, `create_watsonx_llm()`           | `create_ollama_embedding()`, `create_ollama_llm()`                                             |
| `llama-hub`                                                    | Dropped: deprecated, its loaders now ship as `llama-index-readers-*` packages (it was never imported) |
| Python 3.11, `llama-index-core` 0.11, `gradio` 4.44 (plus pins for its web stack) | Python 3.14, current `llama-index-core`, `gradio` 6.30                   |
| Cloud IDE, starter files downloaded with `wget`, `requirements.txt` | A self-contained project: `pyproject.toml`, `Makefile`, its own `.venv` and its own Ollama in `compose.yaml`; every file is also given in full below |
| Proxycurl API key typed into `config.py`                       | Read from `.env`                                                                               |
| Lab code style (`Optional[...]`, `Dict[...]`, mixed quotes)    | Formatted and linted with ruff (`str \| None`, `dict[...]`)                                    |
| App on `0.0.0.0:5000`                                          | App on `127.0.0.1:5000`, analytics off: nothing leaves the machine                             |
| `answer_user_query` retrieved the nodes, built a context string it never used, then let a query engine retrieve them again | It retrieves once and passes those nodes to a response synthesizer |
| `chatbot_interface` and `chat_with_profile` crashed on `.response` when the query had failed (the error path returns a plain string) | They use `str(response)`, which works for both |
| `gr.Chatbot` history as `[user, bot]` pairs                    | `{"role": ..., "content": ...}` messages, the only format Gradio 6 accepts                     |

## A quick look at AI Icebreaker bot

### What Does the AI Icebreaker bot do?

The LinkedIn Icebreaker Bot is an AI-powered tool that generates personalized conversation starters from LinkedIn profiles. Using LlamaIndex and a local Qwen3 model, it extracts profile data (via built-in mock data), processes it through a RAG pipeline, and produces tailored insights about a person's career. With the convenient mock data option, you can demonstrate the bot's capabilities without an API key - perfect for testing or presentations. Simply select "Use Mock Data" in the interface to instantly analyze a pre-loaded professional profile.

Available as both a command-line tool and web interface, the bot helps you make meaningful professional connections by replacing generic small talk with relevant, personalized conversation starters based on someone's actual experience and achievements.

```
icebreaker (CLI: main.py, web: app.py on http://127.0.0.1:5000)
   ├──► mock profile JSON (downloaded once per run)
   ├──► LlamaIndex in-memory vector index
   └──► Ollama (Docker, localhost:11434)
          ├─ qwen3-8b-rag          Qwen3-8B Q8_0, writes the facts and answers
          └─ Qwen3-Embedding-0.6B  turns profile chunks and questions into vectors
```

## Tips for the best experience - please read!

#### Keep the following tips handy and refer to them at any point of confusion throughout the tutorial. Do not worry if they seem irrelevant now. We will go through everything step by step later.

-   Ollama must be running before you start the bot. Check it with `docker compose ps` in this folder. After a reboot, start it again with `make up`.

-   Always run the commands from this project folder (`lessons/module03/04_icebreaker_bot_with_local_qwen3`). The scripts import `config` and `modules` from the current directory, so they fail to run from anywhere else. The `make` targets use the project's `.venv` directly, so they need no activated venv.

-   At the end of each section, you will be given the full file for that part.

-   The first request after Ollama starts takes 15-35 seconds while the models load into memory. After that, answers take a few seconds. The models stay loaded for 30 minutes after their last use.

-   The web application runs on <http://127.0.0.1:5000>. To stop it, hit `Ctrl+C` in the terminal.

-   If you encounter an error after processing a profile or asking a question, read the terminal: every module logs what it is doing. See [Troubleshooting](#troubleshooting) at the end.

## Setting up your development environment

Before we dive into development, let's set up the local models and the project. You need:

-   Docker Engine with the Compose plugin (`docker compose version`)
-   Python 3.14 (the version this lab was tested with)
-   About 10 GB of free disk for the Ollama image and the models
-   Recommended: an NVIDIA GPU with the NVIDIA Container Toolkit installed (Qwen3-8B Q8_0 uses about 11 GB of GPU memory). Without one, everything still runs on the CPU, just more slowly.

* * *

## Step 1: Start the local models with Ollama

This project brings its own Ollama service (`compose.yaml`) and the `Modelfile` for `qwen3-8b-rag`. Run the one-time setup from this folder, while online:

```bash
cd lessons/module03/04_icebreaker_bot_with_local_qwen3
make models
```

`make models` runs these steps (you can also run them by hand):

```bash
# 1. Start Ollama in the background (make up)
docker compose up -d

# 2. Download the models from Hugging Face into the icebreaker_ollama volume
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q8_0               # ~8.7 GB
docker compose exec ollama ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0   # ~0.6 GB

# 3. Build qwen3-8b-rag from the Modelfile (reuses the downloaded weights, no extra disk)
docker compose cp Modelfile ollama:/Modelfile
docker compose exec ollama ollama create qwen3-8b-rag -f /Modelfile

# 4. Check: qwen3-8b-rag and both hf.co/... models
docker compose exec ollama ollama list
```

Step 3 matters: the chat template Ollama generates for the raw `hf.co/Qwen/Qwen3-8B-GGUF` model ignores "thinking off", so Qwen3's reasoning would leak into every fact and answer. The `Modelfile` fixes the template; the bot then asks for thinking off on every request.

The models live in this project's own Docker volume, `icebreaker_ollama`, separate from the module 2 lesson's. Both projects publish Ollama on port 11434, so only one can run at a time: stop the other one with `docker compose down` in its folder first.

No NVIDIA GPU? Delete the `deploy:` block from `compose.yaml` before step 1.

## Step 2: The project structure

This folder already holds the finished project. Read along and run it, or build it yourself: start an empty folder with `pyproject.toml`, `Makefile`, `compose.yaml`, `Modelfile` and `.env.sample` from this one, then create each Python file as you go through the parts below.

```
04_icebreaker_bot_with_local_qwen3/
├── 04_icebreaker_bot_with_local_qwen3.md  # This lab
├── README.md                  # Setup, run, configuration and troubleshooting, in short
├── pyproject.toml             # Dependencies and ruff settings
├── Makefile                   # Shortcuts: setup, models, run, lint
├── compose.yaml               # The Ollama service
├── Modelfile                  # Builds qwen3-8b-rag
├── .env.sample                # Template for .env (Proxycurl API key)
├── config.py                  # Configuration settings
├── test_config.py             # Quick check of config.py
├── modules/
│   ├── __init__.py
│   ├── data_extraction.py     # LinkedIn profile data extraction
│   ├── data_processing.py     # Data splitting and indexing
│   ├── llm_interface.py       # Local LLM and embedding model setup
│   └── query_engine.py        # Query processing and response generation
├── app.py                     # Gradio interface
└── main.py                    # Main script to run without Gradio
```

## Step 3: Install the libraries

The dependencies are declared in `pyproject.toml`:

```toml
[project]
name = "icebreaker-bot"
version = "0.1.0"
description = "LinkedIn icebreaker bot: RAG with LlamaIndex and a local Qwen3 served by Ollama"
requires-python = ">=3.14"
dependencies = [
    "llama-index-core",
    "llama-index-llms-ollama",
    "llama-index-embeddings-ollama",
    "llama-index-readers-web",
    "requests",
    "python-dotenv",
    "gradio==6.30.0",
]

[project.optional-dependencies]
dev = ["ruff"]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["app", "config", "main"]
packages = ["modules"]

[tool.ruff]
line-length = 120
target-version = "py314"

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
]

[tool.ruff.lint.per-file-ignores]
"config.py" = ["E501"]  # the prompt templates are long lines of prose

[tool.ruff.lint.isort]
known-first-party = ["config", "modules"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

This set of packages builds a retrieval-augmented generation (RAG) system on local models. `llama-index-core` provides document indexing, splitting, retrieval and query engines. `llama-index-llms-ollama` and `llama-index-embeddings-ollama` connect LlamaIndex to the Qwen3 language model and the Qwen3 embedding model served by Ollama (they install the `ollama` Python client). `llama-index-readers-web` adds the ability to ingest content directly from websites; this lab keeps it from the original but does not import it (it is also the heaviest dependency: it pulls in Scrapy, Selenium and Playwright). `requests` fetches the mock profile (and calls a profile API, if you plug one in), `python-dotenv` reads the `.env` file, and `gradio` builds the web interface. The `dev` extra adds `ruff`, the linter and formatter behind `make lint`.

The `[tool.setuptools]` table lists the project's modules, so `pip install -e .` installs the dependencies and makes the project importable without copying it.

Create the virtual environment, install everything into it, and create your `.env`. It takes a few minutes:

```bash
make venv      # python -m venv .venv
make install   # pip install -e ".[dev]", then chmod -R go-w .venv
make env       # copy .env.sample to .env (won't overwrite an existing one)
```

The `chmod` is not cosmetic: LlamaIndex keeps its NLTK data (the sentence tokenizer that `SentenceSplitter` uses) inside the venv, and NLTK refuses to read data that other users could write to. With a `umask` of `002`, a plain `pip install` leaves those folders group-writable, and the splitter then silently produces 0 nodes.

Now that your environment is set up, you're ready to start building your AI Icebreaker bot!

## Part 1: Setting up your configuration

In this step, we'll set up the configuration file for our Icebreaker Bot. The `config.py` file centralizes all our settings in one place, making it easier to manage our application parameters.

## Step 1: Create the config.py file

Create `config.py` with the content below:

```python
"""Configuration settings for the Icebreaker Bot."""

import os

from dotenv import load_dotenv

# Read secrets from the .env file next to this project
load_dotenv()

# Local Ollama settings
OLLAMA_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 300.0  # seconds; the first request also loads the model into memory

# Model settings
LLM_MODEL_ID = "qwen3-8b-rag"
EMBEDDING_MODEL_ID = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
CONTEXT_WINDOW = 8192  # tokens Ollama reserves for prompt + answer (num_ctx)

# ProxyCurl API settings
PROXYCURL_API_KEY = os.getenv("PROXYCURL_API_KEY", "")  # Set it in .env

# Mock data URL
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
TOP_K = 50
TOP_P = 1.0

# Node settings
CHUNK_SIZE = 500

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""
```

## Step 2: Understanding the settings

-   `OLLAMA_URL`: Where the Ollama server listens. The `compose.yaml` publishes it on `localhost:11434`.

-   `REQUEST_TIMEOUT`: How long to wait for Ollama, in seconds. It is generous because the first request also loads the model into memory.

-   `LLM_MODEL_ID`: The Ollama model that writes the facts and answers. `qwen3-8b-rag` is Qwen3-8B with the fixed chat template from Step 1 of the setup.

-   `EMBEDDING_MODEL_ID`: The Ollama model that turns text into vectors.

-   `CONTEXT_WINDOW`: How many tokens Ollama reserves for the prompt plus the answer (Ollama calls it `num_ctx`). 5 chunks of up to 500 tokens, the template and a 500-token answer fit in 8192 with room to spare. A bigger window uses more GPU memory.

-   `PROXYCURL_API_KEY`: Read from `.env` by `load_dotenv()`, so the key never sits in the source code. Leave it empty to use the mock data.

-   `MOCK_DATA_URL`: A pre-made LinkedIn profile JSON, used instead of the API.

-   `TEMPERATURE`, `MAX_NEW_TOKENS`, `TOP_K`, `TOP_P`: Default generation settings, explained in Part 5.

## Step 3: Understanding the Prompt Templates

The most important part of this file is the prompt templates that will guide the LLM in generating responses. These templates use placeholders like `{context_str}` and `{query_str}` that will be replaced with actual content during execution.

### Initial Facts Template:

-   Purpose: Generates 3 interesting facts about a person's career or education

-   Context: Uses the `{context_str}` placeholder where LinkedIn profile data will be inserted

-   Instructions: Tells the LLM to answer based only on the provided context

-   Output Format: Requests detailed answers about the person

### User Question Template:

-   Purpose: Answers specific questions about the LinkedIn profile

-   Context: Uses the `{context_str}` placeholder for LinkedIn data

-   Query: Uses the `{query_str}` placeholder for the user's question

-   Instructions: Tells the LLM to answer based only on the context, and to say "I don't know" if the information isn't available

## Step 4: Additional Configuration Options (Optional)

You might want to adjust some of the other settings:

-   `CHUNK_SIZE`: Controls how large each text chunk is, in tokens, when splitting the LinkedIn data. If you want more granular retrieval, you could reduce this (e.g., to 300).

-   `SIMILARITY_TOP_K`: Determines how many similar chunks to retrieve when answering queries. Increasing this could give more comprehensive but potentially noisier responses. Keep `SIMILARITY_TOP_K * CHUNK_SIZE` well under `CONTEXT_WINDOW`.

```python
# Adjust these settings if needed
CHUNK_SIZE = 400  # Smaller chunks for more granular retrieval
SIMILARITY_TOP_K = 7  # Retrieve more chunks for more comprehensive answers
```

### Testing Your Configuration

To verify your configuration is correct, create `test_config.py`:

```python
import config

print("Initial Facts Template defined:", bool(config.INITIAL_FACTS_TEMPLATE))
print("User Question Template defined:", bool(config.USER_QUESTION_TEMPLATE))
print("LLM model:", config.LLM_MODEL_ID)
print("Embedding model:", config.EMBEDDING_MODEL_ID)
print("Chunk Size:", config.CHUNK_SIZE)
print("Similarity Top K:", config.SIMILARITY_TOP_K)
```

Run this script to make sure all your settings are properly defined. You should see the configurations printed in your terminal. If any configuration is incorrect, go back and modify `config.py`.

```bash
make check-config   # .venv/bin/python test_config.py
```

## What We've Accomplished

In this step, we've:

-   Pointed the bot at the local Ollama server and its two models

-   Defined prompt templates that will guide our LLM's responses

-   Configured retrieval parameters for our RAG workflow

-   Created a centralized configuration that all our modules can use

Next, we'll go through the core concepts, then implement the data extraction module that fetches LinkedIn profile data using these settings.

## Part 2: Understanding the Core Concepts

## 1\. Large Language Models and RAG (Optional - Skip if you already understand these concepts)

### What are Large Language Models (LLM)?

[Large language models](https://www.ibm.com/topics/large-language-models) are a type of artificial intelligence (AI) models that are trained on a large corpus of text data. LLMs are designed to generate human-like text responses to a wide range of questions. They are based on the Transformer architecture and are pretrained on a variety of language tasks to improve their performance.

### What is Ollama?

[Ollama](https://ollama.com) is a server that runs open-weight models on your own machine and exposes them over a local HTTP API, on port 11434. It downloads models (here, straight from Hugging Face in the GGUF format), loads them into GPU or CPU memory on the first request, and unloads them after a period of inactivity. In this project it serves two models: `qwen3-8b-rag` for text generation and `Qwen3-Embedding-0.6B` for embeddings. Because everything runs locally, no API key or project ID is needed and no profile data leaves your machine.

### What is Qwen3?

[Qwen3](https://qwenlm.github.io/blog/qwen3/) is a family of open-weight models from Alibaba's Qwen team. We use **Qwen3-8B**, an 8-billion-parameter model, in the `Q8_0` quantization: its weights are stored in 8 bits instead of 16, which roughly halves the memory needed with almost no loss in quality. Qwen3 can "think" (write out its reasoning before answering) or answer directly; for short, grounded answers like icebreakers we turn thinking off. **Qwen3-Embedding-0.6B** is the matching embedding model from the same family.

### What is LlamaIndex?

[LlamaIndex](https://www.ibm.com/topics/llamaindex) is an open-source data orchestration platform for creating large language model (LLM) applications. LlamaIndex is accessible in Python and TypeScript, and it uses a set of tools and features to ease the context augmentation process for [generative AI (genAI)](https://www.ibm.com/topics/generative-ai) use cases via a Retrieval-Augmented (RAG) pipeline. It talks to Ollama through the `llama-index-llms-ollama` and `llama-index-embeddings-ollama` integrations, so switching from a cloud model to a local one only changes the model objects; the rest of the pipeline stays the same.

### What is RAG?

While LLMs are built using extensive datasets, they don't naturally include your specific data. [Retrieval-Augmented Generation (RAG)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) solves this issue by merging your data with the existing knowledge of LLMs. Throughout this guide, you'll frequently see mentions of RAG, as it's a critical technique used in query and chat engines, as well as agents, to boost their performance.

In a RAG setup, your data is first loaded, processed, and indexed for quick retrieval. When a user submits a query, the system searches the index to find the most relevant information from your data. This contextual information is then combined with the user's query and sent to the LLM, which generates a response based on this refined data.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ISxumwmkqbJnsY9KN9PzAw/Rag%20Basics.png" alt="ragbascis" width="90%" />

[source](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)

### What are RAG stages?

In the RAG framework, there are five key stages, though this lab will concentrate on the first four. These stages are essential for most large-scale applications you may develop. They include:

-   **Loading**: This step involves importing your data into the workflow, regardless of the source — such as text files, PDFs, websites, databases, or APIs. LlamaHub provides various connectors to simplify this process.

-   **Splitting**: After loading, data is divided into smaller, manageable chunks to improve the relevance of retrieval and the quality of responses from the LLM. Splitting strategies vary depending on the data type and use case — for example, using fixed-size windows, semantic segmentation, or recursive character splitting for text documents.

-   **Indexing**: In this phase, a data structure is built for efficient querying. For LLMs, this often means creating vector embeddings, numerical representations that capture the meaning of the data, along with metadata strategies to ensure precise and context-aware data retrieval.

    -   An `Index` is a key data structure in **LlamaIndex** that allows us to retrieve the relevant context in response to a user query. Indexing is vital because it enables quick and efficient access to relevant chunks of the data, which makes question-answering both faster and more accurate.

    -   At a high level, `Indexes` are built from the `Nodes` created in the previous splitting stage. These indexes are then used to construct `Retrievers` and `Query Engines`, which power question-answering interactions and conversational experiences based on your data.

    -   **What is vector embedding?** Before we can index the nodes, we need to convert them into vector representations. This process, known as `embedding`, allows the model to better understand the text data and compare its semantic similarity to user queries. Vectors enable efficient search and retrieval operations, as similar pieces of text will have similar vector representations.

    -   To embed the nodes, we'll use the `Qwen3 Embedding model` served by Ollama. Once the data is embedded, we store these vectors in a vector store for retrieval during query processing. Here that is LlamaIndex's in-memory vector store: a single profile is small, so there is no need for a separate database.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/FL6t-jfoJfelsfQdshoB4Q/Vector%20Embedding-1.png" alt="vector-embedding" width="80%" />

[source](https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing/)

-   **Storing**: Once indexing is complete, it's important to save the index and its associated metadata to avoid re-indexing the data later.

-   **Querying**: Based on the indexing strategy, there are various ways to perform queries using LLMs and LlamaIndex structures. This can include sub-queries, multi-step queries, or hybrid techniques.

-   **Evaluation**: A crucial step in any workflow is evaluating the effectiveness of your approach. Evaluation provides objective metrics to measure the accuracy, relevance, and speed of query results when comparing different methods or making adjustments.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/0zwxlxbQ17ID4Nn2iFjn_Q/Rag%20Stages.png" alt="ragstages" width="90%" />

[source](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)

* * *

In our Icebreaker Bot, the RAG workflow looks like this:

-   We load LinkedIn profile data

-   We convert it into chunks and generate embeddings

-   We store these embeddings in a vector store

-   When you ask a question, we retrieve the most relevant pieces of the profile

-   We use an LLM to generate personalized answers based on this specific context

This approach ensures that our responses are grounded in the actual LinkedIn profile data rather than the LLM's general knowledge.

### What is prompt engineering?

[Prompt Engineering](https://www.ibm.com/topics/prompt-engineering) is the process of designing and refining the inputs (prompts) given to language models to get desired outputs. It involves crafting questions, instructions, or context in a way that guides the model to generate accurate, relevant, or creative responses. Good prompt engineering can improve the quality, specificity, and usefulness of the generated text, making it a critical skill for leveraging large language models in various applications like chatbots, content generation, or data analysis.

## 2\. LinkedIn Data Extraction

> Please note that while we present both options for extracting profile data in this step, only the mock data option will work in the current app. Nevertheless, we include the API integration example in case you wish to adapt the code for your own development purposes.

### The Data Extraction Process

When extracting LinkedIn profile data, we're using a REST API to fetch structured information about a user's professional background. The process works as follows:

#### 1\. Prepare the request:

-   Format the LinkedIn profile URL

-   Set up authentication headers with your API key

-   Configure parameters (like what data to include)

#### 2\. Send the API request:

-   Make an HTTP GET request to an API's endpoint

-   Pass the profile URL and other parameters

#### 3\. Process the response:

-   Check if the request was successful (HTTP 200 status)

-   Parse the JSON response data

-   Clean the data by removing empty values and unwanted fields

#### 4\. Structure the data:

-   The resulting JSON contains sections like:

    -   Basic information (name, headline, location)

    -   Work experience

    -   Education history

    -   Skills

    -   Certifications

    -   Accomplishments

The data extraction is the first step in our RAG pipeline, providing the raw material that we'll later split, index, and query.

* * *

## Option for Mock Data vs. Real API Calls

Our application provides two options for working with LinkedIn profile data:

### Using Mock Data:

#### Advantages:

-   No API key required

-   No credit consumption

-   Consistent results for testing

#### How it works:

-   A pre-generated LinkedIn profile JSON is loaded from a URL (so this one step needs an internet connection; everything after it is local)

-   This data is treated the same as if it came from the API

-   All RAG processing steps remain the same

### Using Real API Calls:

#### Advantages:

-   Access to real, up-to-date LinkedIn profiles

-   Ability to analyze any public LinkedIn profile

-   More diverse and personalized results

#### How it works:

-   Your API key authenticates requests

-   The API fetches real-time data from LinkedIn

-   Each request consumes certain amount of credits from your account

### Choosing the Right Approach:

#### Use mock data for:

-   Learning and development

-   Testing your implementation

-   Demonstrations without API costs

#### Use real API calls for:

-   Analyzing specific profiles of interest

-   Production applications

-   Situations where up-to-date data is crucial

In the next parts of this tutorial, we'll implement both approaches, allowing you to switch between them easily based on your needs.

## Part 3: Implementing LinkedIn Profile Data Extraction

> Please note that while we present both options for extracting profile data in this step, as mentioned previously, only the mock data option will work in the current app. Nevertheless, we include the API integration example in case you wish to adapt the code for your own development purposes.

In this step, we'll implement the data extraction module that fetches LinkedIn profile data either through API or by using mock data. This module is unchanged from the original lab: it does not talk to any model.

## Step 1: Create the data\_extraction.py file

Create `modules/data_extraction.py` and start it with the imports and the logger:

```python
"""Module for extracting LinkedIn profile data."""

import logging
import time
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)
```

In this file, we have to implement the `extract_linkedin_profile` function which retrieves and processes LinkedIn profile data, either by fetching it from the API using a provided URL and API key, or by loading pre-configured mock data when the mock option is enabled.

### Exercise: Take a moment to think about how you would implement this function and create the pseudocode for it before moving on to the next step.

## Step 2: Implement the extract\_linkedin\_profile Function

Now, let's implement the function to handle both mock data and real API calls. Add the code below to `modules/data_extraction.py` and save the file.

```python
def extract_linkedin_profile(
    linkedin_profile_url: str, api_key: str | None = None, mock: bool = False
) -> dict[str, Any]:
    """Extract LinkedIn profile data using ProxyCurl API or loads a premade JSON file.

    Args:
        linkedin_profile_url: The LinkedIn profile URL to extract data from.
        api_key: ProxyCurl API key. Required if mock is False.
        mock: If True, loads mock data from a premade JSON file instead of using the API.

    Returns:
        Dictionary containing the LinkedIn profile data.
    """
    start_time = time.time()

    try:
        if mock:
            logger.info("Using mock data from a premade JSON file...")
            mock_url = config.MOCK_DATA_URL
            response = requests.get(mock_url, timeout=30)
        else:
            # Ensure API key is provided when mock is False
            if not api_key:
                raise ValueError("ProxyCurl API key is required when mock is set to False.")

            logger.info("Starting to extract the LinkedIn profile...")

            # Set up the API endpoint and headers
            api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
            headers = {"Authorization": f"Bearer {api_key}"}

            # Prepare parameters for the request
            params = {
                "url": linkedin_profile_url,
                "fallback_to_cache": "on-error",
                "use_cache": "if-present",
                "skills": "include",
                "inferred_salary": "include",
                "personal_email": "include",
                "personal_contact_number": "include",
            }

            logger.info(f"Sending API request to ProxyCurl at {time.time() - start_time:.2f} seconds...")

            # Send API request
            response = requests.get(api_endpoint, headers=headers, params=params, timeout=10)

        logger.info(f"Received response at {time.time() - start_time:.2f} seconds...")

        # Check if response is successful
        if response.status_code == 200:
            try:
                # Parse the JSON response
                data = response.json()

                # Clean the data, remove empty values and unwanted fields
                data = {
                    k: v
                    for k, v in data.items()
                    if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
                }

                # Remove profile picture URLs from groups to clean the data
                if data.get("groups"):
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)

                return data
            except ValueError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response content: {response.text[:200]}...")  # Print first 200 chars
                return {}
        else:
            logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {}

    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}
```

## How this implementation works:

Let's break down what our implementation does:

### Handling Mock Data vs. API Calls

If `mock=True`, we load data from a predefined URL in the config. If `mock=False`, we prepare an API request to the API provider with the provided API key.

### API Request Configuration

-   We set up the API endpoint

-   We configure the authorization header with the API key

-   We specify parameters like:

    -   The LinkedIn profile URL

    -   Cache settings to reduce API usage

    -   Additional data to include (skills, salary info, etc.)

### Response Processing

-   We check for a successful status code (200)

-   We parse the JSON response

-   We clean the data by:

    -   Removing empty values (empty lists, empty strings, None)

    -   Excluding unwanted fields like "people\_also\_viewed"

    -   Removing profile picture URLs from groups

### Error Handling

-   We wrap everything in try/except blocks to gracefully handle errors

-   We log detailed error messages to help debugging

-   We return an empty dictionary if anything goes wrong

<details>
<summary>The full <code>modules/data_extraction.py</code></summary>

```python
"""Module for extracting LinkedIn profile data."""

import logging
import time
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)


def extract_linkedin_profile(
    linkedin_profile_url: str, api_key: str | None = None, mock: bool = False
) -> dict[str, Any]:
    """Extract LinkedIn profile data using ProxyCurl API or loads a premade JSON file.

    Args:
        linkedin_profile_url: The LinkedIn profile URL to extract data from.
        api_key: ProxyCurl API key. Required if mock is False.
        mock: If True, loads mock data from a premade JSON file instead of using the API.

    Returns:
        Dictionary containing the LinkedIn profile data.
    """
    start_time = time.time()

    try:
        if mock:
            logger.info("Using mock data from a premade JSON file...")
            mock_url = config.MOCK_DATA_URL
            response = requests.get(mock_url, timeout=30)
        else:
            # Ensure API key is provided when mock is False
            if not api_key:
                raise ValueError("ProxyCurl API key is required when mock is set to False.")

            logger.info("Starting to extract the LinkedIn profile...")

            # Set up the API endpoint and headers
            api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
            headers = {"Authorization": f"Bearer {api_key}"}

            # Prepare parameters for the request
            params = {
                "url": linkedin_profile_url,
                "fallback_to_cache": "on-error",
                "use_cache": "if-present",
                "skills": "include",
                "inferred_salary": "include",
                "personal_email": "include",
                "personal_contact_number": "include",
            }

            logger.info(f"Sending API request to ProxyCurl at {time.time() - start_time:.2f} seconds...")

            # Send API request
            response = requests.get(api_endpoint, headers=headers, params=params, timeout=10)

        logger.info(f"Received response at {time.time() - start_time:.2f} seconds...")

        # Check if response is successful
        if response.status_code == 200:
            try:
                # Parse the JSON response
                data = response.json()

                # Clean the data, remove empty values and unwanted fields
                data = {
                    k: v
                    for k, v in data.items()
                    if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
                }

                # Remove profile picture URLs from groups to clean the data
                if data.get("groups"):
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)

                return data
            except ValueError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response content: {response.text[:200]}...")  # Print first 200 chars
                return {}
        else:
            logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {}

    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}
```

</details>

## What We've Accomplished

In this step, we've:

-   Implemented the LinkedIn profile data extraction function

-   Added support for both mock data and real API calls

-   Added robust error handling and logging

-   Cleaned the response data for easier processing

This function forms the foundation of our RAG pipeline, providing the raw data that we'll process, index, and query in the subsequent steps.

## Part 4: Implementing Data Processing for RAG

In this step, we'll implement the data processing module that transforms raw LinkedIn profile data into a format suitable for retrieval and question answering. This involves splitting the data into chunks, creating vector embeddings, and storing them in a searchable vector store.

## Step 1: Create the data\_processing.py file

Create `modules/data_processing.py` and start it with the imports and the logger. It imports `create_ollama_embedding`, which you will write in Part 5:

```python
"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

import config
from modules.llm_interface import create_ollama_embedding

logger = logging.getLogger(__name__)
```

We need to implement three functions:

1.  `split_profile_data`: Divides the profile data into manageable chunks

2.  `create_vector_database`: Creates a vector index from the chunks

3.  `verify_embeddings`: Ensures all embeddings were created properly

### Exercise: Take a moment to think about how you would implement these functions and create the pseudocode for them before moving on to the next step.

## Step 2: Implement the split\_profile\_data Function

In the traditional RAG framework, after the Loading stage, the next step is **Splitting**. This involves dividing the data into smaller, manageable chunks that can be efficiently embedded into a vector store and later retrieved for answering user queries.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hfC81XkfbupgVpjT-l0Ybg/RAG-Splitting.png" alt="ragsplitting" width="80%" />

[source](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)

First, let's implement the function to split the profile data into nodes. Add the code below to `modules/data_processing.py`.

```python
def split_profile_data(profile_data: dict[str, Any]) -> list:
    """Splits the LinkedIn profile JSON data into nodes.

    Args:
        profile_data: LinkedIn profile data dictionary.

    Returns:
        List of document nodes.
    """
    try:
        # Convert the profile data to a JSON string
        profile_json = json.dumps(profile_data)

        # Create a Document object from the JSON string
        document = Document(text=profile_json)

        # Split the document into nodes using SentenceSplitter
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
        nodes = splitter.get_nodes_from_documents([document])

        logger.info(f"Created {len(nodes)} nodes from profile data")
        return nodes
    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}")
        return []
```

This function:

-   Converts the profile data dictionary to a JSON string

-   Creates a Document object from the string

-   Uses a SentenceSplitter to divide the text into chunks of size specified in config

-   Returns the resulting nodes

In our project, we need to split the LinkedIn profile data into smaller segments or nodes. These nodes represent logical chunks of information that can later be queried.

Since the LinkedIn profile data is a JSON file, we first convert it into a textual format, and then split the text into smaller nodes of up to 500 tokens each (`SentenceSplitter` measures `chunk_size` in tokens, not characters). The mock profile becomes 17 nodes.

This splitting process ensures that our data is indexed in manageable pieces, making it easier for the model to retrieve relevant information based on user queries.

## Step 3: Implement the create\_vector\_database Function

In this step, we move to **Indexing** and **Storing** these nodes for efficient retrieval. This step is critical for building a foundation for RAG, where we retrieve relevant data to answer user queries.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/AEEhG4hgyHo0s69ZmGd9hQ/RAG-Index-Storing.png" alt="rag-index-store" width="80%" />

[source](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)

Next, let's implement the function to create the vector index. Add the code below to `modules/data_processing.py`.

```python
def create_vector_database(nodes: list) -> VectorStoreIndex | None:
    """Stores the document chunks (nodes) in a vector database.

    Args:
        nodes: List of document nodes to be indexed.

    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    try:
        # Get the embedding model
        embedding_model = create_ollama_embedding()

        # Create a VectorStoreIndex from the nodes
        index = VectorStoreIndex(nodes=nodes, embed_model=embedding_model, show_progress=True)

        logger.info("Vector database created successfully")
        return index
    except Exception as e:
        logger.error(f"Error in create_vector_database: {e}")
        return None
```

This function:

-   Gets the local embedding model from the `llm_interface` module

-   Creates a `VectorStoreIndex` with the nodes and embedding model. It sends each node to Ollama to be embedded and keeps the vectors in memory

-   Returns the index or None if an error occurs

## Step 4: Implement the verify\_embeddings Function

Finally, let's implement the function to verify the embeddings. Add the code below to `modules/data_processing.py`.

```python
def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have been properly embedded.

    Args:
        index: VectorStoreIndex to verify.

    Returns:
        True if all embeddings are valid, False otherwise.
    """
    try:
        vector_store = index._storage_context.vector_store
        node_ids = list(index.index_struct.nodes_dict.keys())
        missing_embeddings = False

        for node_id in node_ids:
            embedding = vector_store.get(node_id)
            if embedding is None:
                logger.warning(f"Node ID {node_id} has a None embedding.")
                missing_embeddings = True
            else:
                logger.debug(f"Node ID {node_id} has a valid embedding.")

        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False
        else:
            logger.info("All node embeddings are valid")
            return True
    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False
```

This function:

-   Gets the vector store from the index

-   Gets the list of node IDs

-   Checks if each node has a valid embedding

-   Returns True if all embeddings are valid, False otherwise

<details>
<summary>The full <code>modules/data_processing.py</code></summary>

```python
"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

import config
from modules.llm_interface import create_ollama_embedding

logger = logging.getLogger(__name__)


def split_profile_data(profile_data: dict[str, Any]) -> list:
    """Splits the LinkedIn profile JSON data into nodes.

    Args:
        profile_data: LinkedIn profile data dictionary.

    Returns:
        List of document nodes.
    """
    try:
        # Convert the profile data to a JSON string
        profile_json = json.dumps(profile_data)

        # Create a Document object from the JSON string
        document = Document(text=profile_json)

        # Split the document into nodes using SentenceSplitter
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
        nodes = splitter.get_nodes_from_documents([document])

        logger.info(f"Created {len(nodes)} nodes from profile data")
        return nodes
    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}")
        return []


def create_vector_database(nodes: list) -> VectorStoreIndex | None:
    """Stores the document chunks (nodes) in a vector database.

    Args:
        nodes: List of document nodes to be indexed.

    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    try:
        # Get the embedding model
        embedding_model = create_ollama_embedding()

        # Create a VectorStoreIndex from the nodes
        index = VectorStoreIndex(nodes=nodes, embed_model=embedding_model, show_progress=True)

        logger.info("Vector database created successfully")
        return index
    except Exception as e:
        logger.error(f"Error in create_vector_database: {e}")
        return None


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have been properly embedded.

    Args:
        index: VectorStoreIndex to verify.

    Returns:
        True if all embeddings are valid, False otherwise.
    """
    try:
        vector_store = index._storage_context.vector_store
        node_ids = list(index.index_struct.nodes_dict.keys())
        missing_embeddings = False

        for node_id in node_ids:
            embedding = vector_store.get(node_id)
            if embedding is None:
                logger.warning(f"Node ID {node_id} has a None embedding.")
                missing_embeddings = True
            else:
                logger.debug(f"Node ID {node_id} has a valid embedding.")

        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False
        else:
            logger.info("All node embeddings are valid")
            return True
    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False
```

</details>

## What We've Accomplished

In this step, we've:

-   Implemented the function to split LinkedIn profile data into manageable chunks

-   Implemented the function to create a vector index from these chunks

-   Implemented the function to verify the embeddings

These functions form the core of the indexing phase of our RAG pipeline. With the data now properly processed and indexed, we can move on to the models that power it.

## Part 5: Implementing the interface to the local Qwen3 language and embedding models

In this section, we'll implement the interface to the language and embedding models that Ollama serves locally. This is a crucial component of our Icebreaker Bot as it handles the connection to the AI models that power our application.

## Step 1: Create the llm\_interface.py file

Create `modules/llm_interface.py` and start it with the imports and the logger:

```python
"""Module for interfacing with the local Qwen3 models served by Ollama."""

import logging

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import config

logger = logging.getLogger(__name__)
```

In this file, we have to implement 3 functions:

-   `create_ollama_embedding()`: This function creates the embedding model that converts text into vector representations. It should return an `OllamaEmbedding` instance configured with the model name and the Ollama URL from our config file. This embedding model will be used to transform LinkedIn profile data chunks into vectors for semantic search.

-   `create_ollama_llm()`: This function creates the language model that generates responses to user queries. It should return an `Ollama` LLM instance configured with parameters that control the generation process, such as temperature (for randomness), token limits, and decoding methods. This LLM will be responsible for generating interesting facts and answering questions about LinkedIn profiles.

-   `change_llm_model()`: This utility function allows us to dynamically switch between different language models at runtime. It should update the LLM model ID in our config and log the change. This flexibility enables experimenting with different models to compare their performance on icebreaker generation tasks.

### Exercise: Take a moment to think about how you would implement these functions and create the pseudocode for them before moving on to the next step.

## Step 2: Implement the Embedding Model Function

Next, we'll implement the function that creates our embedding model. Add the code below to `modules/llm_interface.py`.

```python
def create_ollama_embedding() -> OllamaEmbedding:
    """Creates a local Qwen3 Embedding model, served by Ollama, for vector representation.

    Returns:
        OllamaEmbedding model.
    """
    ollama_embedding = OllamaEmbedding(
        model_name=config.EMBEDDING_MODEL_ID,
        base_url=config.OLLAMA_URL,
    )
    logger.info(f"Created Ollama Embedding model: {config.EMBEDDING_MODEL_ID}")
    return ollama_embedding
```

This function creates an instance of `OllamaEmbedding` which we'll use to convert text chunks from LinkedIn profiles into vector representations. These vectors capture the semantic meaning of the text, allowing us to find the most relevant information when answering user queries.

**Key Parameters:**

-   `model_name`: Specifies which embedding model to use (defined in config.py). It must match a model listed by `ollama list`

-   `base_url`: The URL of the Ollama server

There is no project ID or API key: the model runs on your machine. The original lab also passed `truncate_input_tokens`; `OllamaEmbedding` has no such parameter and does not need one, because every chunk (at most 500 tokens) is far below the embedding model's input limit.

## Step 3: Implement the Language Model Function

Now, let's implement the function that creates our language model. Add the code below to `modules/llm_interface.py`.

```python
def create_ollama_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample",
) -> Ollama:
    """Creates a local Qwen3 LLM, served by Ollama, for generating responses.

    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use (sample, greedy).

    Returns:
        Ollama LLM.
    """
    additional_kwargs = {
        "num_predict": max_new_tokens,
        "top_k": 1 if decoding_method == "greedy" else config.TOP_K,
        "top_p": config.TOP_P,
    }

    ollama_llm = Ollama(
        model=config.LLM_MODEL_ID,
        base_url=config.OLLAMA_URL,
        temperature=temperature,
        context_window=config.CONTEXT_WINDOW,
        request_timeout=config.REQUEST_TIMEOUT,
        thinking=False,
        additional_kwargs=additional_kwargs,
    )
    logger.info(f"Created Ollama LLM: {config.LLM_MODEL_ID}")
    return ollama_llm
```

This function creates an instance of `Ollama`, LlamaIndex's client for language models served by Ollama, that we'll use to generate responses to user queries. It configures the local Qwen3 model with the appropriate parameters for our use case.

**Key Parameters:**

-   `temperature`: Controls the randomness of the generated text. Lower values (like 0.0) make responses more deterministic and focused, while higher values introduce more creativity and variation.

-   `max_new_tokens`: Limits the length of the generated response. Ollama calls this option `num_predict`.

-   `decoding_method`: Determines how the model selects the next token. Ollama has no option with this name, so the function maps it onto the sampling settings:

    -   "sample": Samples from the `top_k` most likely tokens (more creative)

    -   "greedy": Sets `top_k` to 1, so the most likely next token is always chosen (more deterministic)

-   `additional_kwargs`: Extra Ollama options passed with every request:

    -   `num_predict`: Maximum number of tokens to generate

    -   `top_k`: Limits token selection to the top k most probable tokens

    -   `top_p`: Uses nucleus sampling to select from tokens that comprise the top p probability mass

-   `context_window`: The `num_ctx` Ollama allocates for this model; LlamaIndex also uses it to know how much retrieved text fits in a prompt.

-   `thinking=False`: Asks Qwen3 to answer directly, without writing out its reasoning first. This only works with `qwen3-8b-rag`, whose chat template honors it.

-   `request_timeout`: How long to wait for Ollama before giving up.

The original lab also set `min_new_tokens`; Ollama has no equivalent option, so it is gone.

## Step 4: Implement Model Switching Functionality

Finally, we'll implement a function that allows us to switch between different LLM models. Add the code below to `modules/llm_interface.py`.

```python
def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.

    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
```

This function allows users to experiment with different language models. It updates the `LLM_MODEL_ID` in the config module to use a different model for generating responses. In this project, we only use `qwen3-8b-rag`, but any model you pull into Ollama can be used: pull it with `docker compose exec ollama ollama pull <model>`, then add its name to the model list in `app.py` (Part 8) or pass it to `main.py` with `--model`. See the [Ollama library](https://ollama.com/library) for available models.

<details>
<summary>The full <code>modules/llm_interface.py</code></summary>

```python
"""Module for interfacing with the local Qwen3 models served by Ollama."""

import logging

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import config

logger = logging.getLogger(__name__)


def create_ollama_embedding() -> OllamaEmbedding:
    """Creates a local Qwen3 Embedding model, served by Ollama, for vector representation.

    Returns:
        OllamaEmbedding model.
    """
    ollama_embedding = OllamaEmbedding(
        model_name=config.EMBEDDING_MODEL_ID,
        base_url=config.OLLAMA_URL,
    )
    logger.info(f"Created Ollama Embedding model: {config.EMBEDDING_MODEL_ID}")
    return ollama_embedding


def create_ollama_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample",
) -> Ollama:
    """Creates a local Qwen3 LLM, served by Ollama, for generating responses.

    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use (sample, greedy).

    Returns:
        Ollama LLM.
    """
    additional_kwargs = {
        "num_predict": max_new_tokens,
        "top_k": 1 if decoding_method == "greedy" else config.TOP_K,
        "top_p": config.TOP_P,
    }

    ollama_llm = Ollama(
        model=config.LLM_MODEL_ID,
        base_url=config.OLLAMA_URL,
        temperature=temperature,
        context_window=config.CONTEXT_WINDOW,
        request_timeout=config.REQUEST_TIMEOUT,
        thinking=False,
        additional_kwargs=additional_kwargs,
    )
    logger.info(f"Created Ollama LLM: {config.LLM_MODEL_ID}")
    return ollama_llm


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.

    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
```

</details>

## What We've Accomplished

In this step, we've:

-   Set up the interface to both the local embedding model and the local language model

-   Configured key generation parameters including temperature and decoding methods

-   Implemented a model switching function for easy experimentation

-   Created a clean abstraction layer that hides the details of connecting to Ollama

This LLM interface serves as the intelligence layer of our Icebreaker Bot, enabling both the semantic understanding of LinkedIn profiles through embeddings and the generation of personalized icebreakers through the language model.

## Part 6: Creating the query engine

Now that we have our LinkedIn data indexed and our LLM interface configured, we need to create the query engine that will power our Icebreaker Bot's ability to generate insights and answer questions. This module is responsible for retrieving relevant information from our vector index and formatting it into engaging responses.

## What is a Query Engine?

In the context of our RAG (Retrieval-Augmented Generation) system, a query engine is responsible for:

-   Retrieving relevant information from our vector index based on a query

-   Formatting the retrieved information into a prompt for the LLM

-   Generating coherent and accurate responses based on the retrieved context

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4hambo2ZiF_HcZ5M9Uxetg/Index%20Retrieval.png" alt="basic-retrieval-system" width="80%" />

[source](https://docs.llamaindex.ai/en/stable/examples/low_level/retrieval/)

Let's implement our query engine step by step.

## Step 1: Create the query\_engine.py file

Create `modules/query_engine.py` and start it with the imports and the logger:

```python
"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any

from llama_index.core import PromptTemplate, VectorStoreIndex, get_response_synthesizer

import config
from modules.llm_interface import create_ollama_llm

logger = logging.getLogger(__name__)
```

In this file, we have to implement 2 functions:

-   `generate_initial_facts(index)`: This function creates engaging conversation starters based on a person's LinkedIn profile. It should create a local LLM with appropriate parameters for fact generation, construct a prompt template using our predefined template from the config, build a query engine that combines the index with the LLM and prompt, and then execute a query to generate three interesting facts about the person.

-   `answer_user_query(index, user_query)`: This function powers the interactive Q&A capability of our bot. It should create a local LLM optimized for question answering (using different parameters than fact generation), retrieve the most relevant nodes from our vector index based on the user's query, and have the LLM answer from those nodes using our question-answering prompt template. This function implements the complete RAG (Retrieval-Augmented Generation) workflow, with each step visible, that enables accurate, context-aware responses about the LinkedIn profile.

### Exercise: Take a moment to think about how you would implement these functions and create the pseudocode for them before moving on to the next step.

## Step 2: Implement the Fact Generation Function

First, let's implement the function that generates interesting facts about a person based on their LinkedIn profile. Add the code below to `modules/query_engine.py`.

```python
def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person's career or education.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.

    Returns:
        String containing interesting facts about the person.
    """
    try:
        # Create LLM for generating facts
        ollama_llm = create_ollama_llm(temperature=0.0, max_new_tokens=500, decoding_method="sample")

        # Create prompt template
        facts_prompt = PromptTemplate(template=config.INITIAL_FACTS_TEMPLATE)

        # Create query engine
        query_engine = index.as_query_engine(
            streaming=False, similarity_top_k=config.SIMILARITY_TOP_K, llm=ollama_llm, text_qa_template=facts_prompt
        )

        # Execute the query
        query = "Provide three interesting facts about this person's career or education."
        response = query_engine.query(query)

        # Return the facts
        return response.response
    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."
```

### **How This Function Works:**

**1\. LLM Configuration:** We create a language model instance with specific parameters:

-   `temperature=0.0`: Makes the output more deterministic and focused on factual information

-   `max_new_tokens=500`: Allows for a detailed response

-   `decoding_method="sample"`: Samples from the top tokens; with a temperature of 0.0 the result is still effectively deterministic. Raise the temperature (Qwen suggests 0.7 when thinking is off) for more varied icebreakers

**2\. Prompt Template:** We use a predefined template (from config.py) that instructs the LLM how to generate facts based on the profile data.

**3\. Query Engine Creation:** We create a query engine from our index with these parameters:

-   `streaming=False`: Wait for the full response rather than streaming it

-   `similarity_top_k=config.SIMILARITY_TOP_K`: Retrieve the most relevant chunks from the profile

-   `llm=ollama_llm`: Use our configured language model

-   `text_qa_template=facts_prompt`: Use our fact generation prompt template

**4\. Query Execution:** We send a simple query to the engine and return the response

**5\. Error Handling:** We catch any exceptions and return a friendly error message

## Step 3: Implement the Question Answering Function

Now, let's implement the function that answers specific user questions about the LinkedIn profile. Add the code below to `modules/query_engine.py`.

```python
def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answers the user's question using the vector database and the LLM.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        user_query: The user's question.

    Returns:
        Response object containing the answer to the user's question.
    """
    try:
        # Create LLM for answering questions
        ollama_llm = create_ollama_llm(temperature=0.0, max_new_tokens=250, decoding_method="greedy")

        # Create prompt template
        question_prompt = PromptTemplate(template=config.USER_QUESTION_TEMPLATE)

        # Retrieve relevant nodes
        base_retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        source_nodes = base_retriever.retrieve(user_query)

        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(llm=ollama_llm, text_qa_template=question_prompt)

        # Generate the answer from the retrieved nodes
        answer = response_synthesizer.synthesize(user_query, nodes=source_nodes)
        return answer
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."
```

### **How This Function Works:**

**1\. LLM Configuration:** We create a language model instance with specific parameters:

-   `temperature=0.0`: Makes the output more deterministic and focused on factual information

-   `max_new_tokens=250`: Keeps answers concise

-   `decoding_method="greedy"`: Always selects the most likely next token for maximum accuracy

**2\. Prompt Template:** We use a question-answering template that instructs the LLM to answer based on the provided context.

**3\. Relevant Node Retrieval:** This is a key step where we:

-   Create a retriever from our index

-   Embed the user's query with the same Qwen3 embedding model and retrieve the most similar nodes

**4\. Response Synthesis:** A response synthesizer builds the context string from the retrieved nodes, puts it into `{context_str}` (and the question into `{query_str}`), and sends the prompt to the LLM. This is the step a query engine performs after retrieving; doing the two steps separately makes the RAG workflow visible. (The original lab built the context string by hand but never used it, then created a query engine that retrieved the same nodes a second time.)

**5\. Query Execution:** The synthesizer returns a `Response` object: its `.response` holds the answer text, and its `.source_nodes` the chunks it was based on

**6\. Error Handling:** We catch any exceptions and return a friendly error message

<details>
<summary>The full <code>modules/query_engine.py</code></summary>

```python
"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any

from llama_index.core import PromptTemplate, VectorStoreIndex, get_response_synthesizer

import config
from modules.llm_interface import create_ollama_llm

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person's career or education.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.

    Returns:
        String containing interesting facts about the person.
    """
    try:
        # Create LLM for generating facts
        ollama_llm = create_ollama_llm(temperature=0.0, max_new_tokens=500, decoding_method="sample")

        # Create prompt template
        facts_prompt = PromptTemplate(template=config.INITIAL_FACTS_TEMPLATE)

        # Create query engine
        query_engine = index.as_query_engine(
            streaming=False, similarity_top_k=config.SIMILARITY_TOP_K, llm=ollama_llm, text_qa_template=facts_prompt
        )

        # Execute the query
        query = "Provide three interesting facts about this person's career or education."
        response = query_engine.query(query)

        # Return the facts
        return response.response
    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answers the user's question using the vector database and the LLM.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        user_query: The user's question.

    Returns:
        Response object containing the answer to the user's question.
    """
    try:
        # Create LLM for answering questions
        ollama_llm = create_ollama_llm(temperature=0.0, max_new_tokens=250, decoding_method="greedy")

        # Create prompt template
        question_prompt = PromptTemplate(template=config.USER_QUESTION_TEMPLATE)

        # Retrieve relevant nodes
        base_retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        source_nodes = base_retriever.retrieve(user_query)

        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(llm=ollama_llm, text_qa_template=question_prompt)

        # Generate the answer from the retrieved nodes
        answer = response_synthesizer.synthesize(user_query, nodes=source_nodes)
        return answer
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."
```

</details>

## What We've Accomplished

In this step, we've:

-   Implemented the core query engine for our Icebreaker Bot

-   Created functions for generating interesting facts and answering specific questions

-   Set up the RAG pipeline to retrieve relevant context before generating responses

-   Added robust error handling and logging

-   Configured different LLM parameters for different types of queries

This query engine is the heart of our application, connecting the indexed LinkedIn data with the LLM to generate personalized, accurate responses. In the next section, we'll integrate these components into a complete application with a user interface.

## Part 7: Building command-line interface (CLI) application

> Please note that while we present both options for extracting profile data in this step, as mentioned previously, only the mock data option will work in the current app. Nevertheless, we include the API integration example in case you wish to adapt the code for your own development purposes.

Now that we've built all the key components of our Icebreaker Bot, it's time to bring everything together in a command-line interface (CLI) application. This will allow users to interact with the bot directly from the terminal, providing a LinkedIn URL and asking questions about the profile.

## Step 1: Create the modules package file

All four modules exist now, so create `modules/__init__.py` to expose their functions from the `modules` package:

```python
"""
Icebreaker Bot modules package.

This package contains the following modules:
- data_extraction: Functions for extracting LinkedIn profile data
- data_processing: Functions for processing and indexing LinkedIn profile data
- llm_interface: Functions for interfacing with the local Ollama models
- query_engine: Functions for querying indexed LinkedIn profile data
"""

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import create_vector_database, split_profile_data, verify_embeddings
from modules.llm_interface import change_llm_model, create_ollama_embedding, create_ollama_llm
from modules.query_engine import answer_user_query, generate_initial_facts

__all__ = [
    "extract_linkedin_profile",
    "split_profile_data",
    "create_vector_database",
    "verify_embeddings",
    "create_ollama_embedding",
    "create_ollama_llm",
    "change_llm_model",
    "generate_initial_facts",
    "answer_user_query",
]
```

## Step 2: Create the main.py file

Create `main.py` and start it with the imports and the logging setup:

```python
"""Main script for running the Icebreaker Bot."""

import argparse
import logging
import sys
import time

import config
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import create_vector_database, split_profile_data, verify_embeddings
from modules.query_engine import answer_user_query, generate_initial_facts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

logger = logging.getLogger(__name__)
```

In the main.py module, we need to implement two central functions that orchestrate the entire Icebreaker Bot workflow and provide the user interface, plus the `main()` entry point:

-   `process_linkedin(linkedin_url, api_key, mock)`: This function serves as the core orchestrator of our application. It should coordinate the entire RAG pipeline by first extracting LinkedIn profile data (either via API or from mock data), then splitting this data into manageable nodes, creating a vector index from these nodes, verifying that embeddings were properly created, generating initial conversation starters about the person, and finally launching the interactive chatbot interface. This function handles the end-to-end workflow from raw LinkedIn URL to a fully functional conversational assistant.

-   `chatbot_interface(index)`: This function creates a user-friendly command-line interface for interacting with our bot. It should display clear instructions to the user, create a loop that processes user questions about the LinkedIn profile, retrieve answers using our query engine, display these answers in a conversational format, and provide a clean exit mechanism.

### Exercise: Take a moment to think about how you would implement these functions and create the pseudocode for them before moving on to the next step.

## Step 3: Implement the Process LinkedIn Function

This is the core function that orchestrates the entire workflow. Add the code below to `main.py`.

```python
def process_linkedin(linkedin_url, api_key=None, mock=False):
    """
    Processes a LinkedIn URL, extracts data from the profile, and interacts with the user.

    Args:
        linkedin_url: The LinkedIn profile URL to extract or load mock data from.
        api_key: ProxyCurl API key. Required if mock is False.
        mock: If True, loads mock data from a premade JSON file instead of using the API.
    """
    try:
        # Extract the profile data
        profile_data = extract_linkedin_profile(linkedin_url, api_key, mock=mock)

        if not profile_data:
            logger.error("Failed to retrieve profile data.")
            return

        # Split the data into nodes
        nodes = split_profile_data(profile_data)

        # Store in vector database
        vectordb_index = create_vector_database(nodes)

        if not vectordb_index:
            logger.error("Failed to create vector database.")
            return

        # Verify embeddings
        if not verify_embeddings(vectordb_index):
            logger.warning("Some embeddings may be missing or invalid.")

        # Generate and display the initial facts
        initial_facts = generate_initial_facts(vectordb_index)

        print("\nHere are 3 interesting facts about this person:")
        print(initial_facts)

        # Start the chatbot interface
        chatbot_interface(vectordb_index)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
```

### **How This Function Works:**

**1\. Extraction:** It calls our extract\_linkedin\_profile function to get the profile data **2\. Processing:** It splits the data into nodes and creates a vector index **3\. Initial Facts:** It generates 3 interesting facts as a starting point **4\. User Interface:** It prints the facts and launches the chatbot interface **5\. Logging:** Each module logs its step, so you can follow the progress in the terminal **6\. Error Handling:** It catches and reports any errors that occur during processing

## Step 4: Implement the Chatbot Interface

Next, let's create a simple text-based interface for interacting with the bot. Add the code below to `main.py`.

```python
def chatbot_interface(index):
    """
    Provides a simple chatbot interface for user interaction.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
    """
    print("\nYou can now ask more in-depth questions about this person. Type 'exit', 'quit', or 'bye' to quit.")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        print("Bot is typing...", end="")
        sys.stdout.flush()
        time.sleep(1)  # Simulate typing delay
        print("\r", end="")

        response = answer_user_query(index, user_query)
        print(f"Bot: {str(response).strip()}\n")
```

### **How This Function Works:**

**1\. User Input:** It prompts the user for questions about the profile **2\. Query Processing:** It sends user queries to our answer\_user\_query function **3\. Response Display:** It shows the bot's responses in a conversational format. `str(response)` gives the answer text of a `Response` object, and also works when `answer_user_query` returned its error string **4\. Exit Commands:** It allows the user to end the conversation with simple commands

## Step 5: Create the Main Entry Point

Finally, let's create the main function that ties everything together. Add the code below to `main.py`.

```python
def main():
    """Main function to run the Icebreaker Bot."""
    parser = argparse.ArgumentParser(description="Icebreaker Bot - LinkedIn Profile Analyzer")
    parser.add_argument("--url", type=str, help="LinkedIn profile URL")
    parser.add_argument("--api-key", type=str, help="ProxyCurl API key")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of API")
    parser.add_argument("--model", type=str, help='Ollama model to use (e.g., "qwen3-8b-rag")')

    args = parser.parse_args()

    # Use command line arguments or prompt user for input
    linkedin_url = args.url or input("Enter LinkedIn profile URL (or press Enter to use mock data): ")
    use_mock = args.mock or not linkedin_url

    if args.model:
        from modules.llm_interface import change_llm_model

        change_llm_model(args.model)

    api_key = args.api_key or config.PROXYCURL_API_KEY

    if not use_mock and not api_key:
        api_key = input("Enter ProxyCurl API key: ")

    # Use a default URL for mock data if none provided
    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

    process_linkedin(linkedin_url, api_key, mock=use_mock)


if __name__ == "__main__":
    main()
```

### **How This Function Works:**

**1\. Argument Parsing:** It processes command-line arguments **2\. Interactive Prompts:** If required information is missing, it prompts the user **3\. Profile Processing:** It calls our process\_linkedin function with the provided arguments

<details>
<summary>The full <code>main.py</code></summary>

```python
"""Main script for running the Icebreaker Bot."""

import argparse
import logging
import sys
import time

import config
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import create_vector_database, split_profile_data, verify_embeddings
from modules.query_engine import answer_user_query, generate_initial_facts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

logger = logging.getLogger(__name__)


def process_linkedin(linkedin_url, api_key=None, mock=False):
    """
    Processes a LinkedIn URL, extracts data from the profile, and interacts with the user.

    Args:
        linkedin_url: The LinkedIn profile URL to extract or load mock data from.
        api_key: ProxyCurl API key. Required if mock is False.
        mock: If True, loads mock data from a premade JSON file instead of using the API.
    """
    try:
        # Extract the profile data
        profile_data = extract_linkedin_profile(linkedin_url, api_key, mock=mock)

        if not profile_data:
            logger.error("Failed to retrieve profile data.")
            return

        # Split the data into nodes
        nodes = split_profile_data(profile_data)

        # Store in vector database
        vectordb_index = create_vector_database(nodes)

        if not vectordb_index:
            logger.error("Failed to create vector database.")
            return

        # Verify embeddings
        if not verify_embeddings(vectordb_index):
            logger.warning("Some embeddings may be missing or invalid.")

        # Generate and display the initial facts
        initial_facts = generate_initial_facts(vectordb_index)

        print("\nHere are 3 interesting facts about this person:")
        print(initial_facts)

        # Start the chatbot interface
        chatbot_interface(vectordb_index)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")


def chatbot_interface(index):
    """
    Provides a simple chatbot interface for user interaction.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
    """
    print("\nYou can now ask more in-depth questions about this person. Type 'exit', 'quit', or 'bye' to quit.")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        print("Bot is typing...", end="")
        sys.stdout.flush()
        time.sleep(1)  # Simulate typing delay
        print("\r", end="")

        response = answer_user_query(index, user_query)
        print(f"Bot: {str(response).strip()}\n")


def main():
    """Main function to run the Icebreaker Bot."""
    parser = argparse.ArgumentParser(description="Icebreaker Bot - LinkedIn Profile Analyzer")
    parser.add_argument("--url", type=str, help="LinkedIn profile URL")
    parser.add_argument("--api-key", type=str, help="ProxyCurl API key")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of API")
    parser.add_argument("--model", type=str, help='Ollama model to use (e.g., "qwen3-8b-rag")')

    args = parser.parse_args()

    # Use command line arguments or prompt user for input
    linkedin_url = args.url or input("Enter LinkedIn profile URL (or press Enter to use mock data): ")
    use_mock = args.mock or not linkedin_url

    if args.model:
        from modules.llm_interface import change_llm_model

        change_llm_model(args.model)

    api_key = args.api_key or config.PROXYCURL_API_KEY

    if not use_mock and not api_key:
        api_key = input("Enter ProxyCurl API key: ")

    # Use a default URL for mock data if none provided
    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

    process_linkedin(linkedin_url, api_key, mock=use_mock)


if __name__ == "__main__":
    main()
```

</details>

## Testing the Complete CLI Workflow

> Please note that while we present both options for extracting profile data in this step, as mentioned previously, only the mock data option will work in the current app. Nevertheless, we include the API integration example in case you wish to adapt the code for your own development purposes.

To test our CLI application, make sure Ollama is running, then run it with various combinations of arguments (with the venv active, `source .venv/bin/activate`, or with `.venv/bin/python` instead of `python`):

### 1\. With mock data:

```bash
make cli   # python main.py --mock
```

`main()` still asks for a LinkedIn URL when you pass no `--url`; press Enter to go on with the mock profile. You will see the profile being split into nodes, the nodes being embedded, then the three facts and a `You:` prompt. Try questions like "Where did he study?" or "What is his current job?", and a question the profile cannot answer, which should get "I don't know".

### 2\. With a specific model:

```bash
python main.py --mock --model qwen3-8b-rag
```

### 3\. With a LinkedIn URL and API key (replace with a real LinkedIn URL and your API key; this needs a working profile API, see the disclaimer):

```bash
python main.py --url https://www.linkedin.com/in/johndoe/ --api-key YOUR_API_KEY
```

### 4\. Interactively (with prompts):

```bash
python main.py
```

## What We've Accomplished

In this step, we've:

-   Implemented a comprehensive main application that orchestrates all our modules

-   Created a function that processes LinkedIn profiles from start to finish

-   Built a simple but effective chatbot interface for user interaction

-   Added command-line argument parsing for flexible usage

-   Included logging for better debugging and monitoring

-   Added robust error handling throughout the application

The main.py file serves as the glue that connects all our components, providing a seamless user experience from profile URL to personalized icebreakers and conversations.

## Next Steps

With our CLI application complete, users can now interact with the Icebreaker Bot directly from the terminal. However, a web-based interface would make the bot even more accessible and user-friendly. In the next section, we'll build a Gradio web interface that provides all the same functionality with a graphical user interface.

## Part 8: Building a user-friendly web interface with Gradio

Now that we've built our command-line interface, let's create a more user-friendly web interface using Gradio. This will make our Icebreaker Bot accessible to users who prefer graphical interfaces and provide a more polished experience.

> Please note that while we present both options for extracting profile data in this step, as mentioned previously, only the mock data option will work in the current app. Nevertheless, we include the API integration example in case you wish to adapt the code for your own development purposes.

The app.py module creates a web-based interface for our Icebreaker Bot using Gradio, a Python library for building web interfaces for machine learning models. This interface will:

-   Allow users to input LinkedIn profile URLs

-   Process profiles and display interesting facts

-   Provide a chat interface for asking questions about the profile

-   Support switching between different LLM models

Let's implement this step by step.

## Step 1: Import the Necessary Libraries and Our Modules

Create `app.py` and start it with:

```python
"""Gradio web interface for the Icebreaker Bot."""

import logging
import sys
import uuid

import gradio as gr

import config
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import create_vector_database, split_profile_data, verify_embeddings
from modules.llm_interface import change_llm_model
from modules.query_engine import answer_user_query, generate_initial_facts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

# Dictionary to store active conversations
active_indices = {}
```

These imports give us access to:

-   Standard Python libraries for unique IDs and logging

-   Gradio for building the web interface

-   Our custom modules for the core functionality

-   A dictionary to store active conversations across sessions

## Step 2: Implement the Profile Processing Function

Next, let's implement the function that processes LinkedIn profiles. Add the code below to `app.py`.

```python
def process_profile(linkedin_url, api_key, use_mock, selected_model):
    """Process a LinkedIn profile and generate initial facts.

    Args:
        linkedin_url: LinkedIn profile URL to process.
        api_key: ProxyCurl API key.
        use_mock: Whether to use mock data.
        selected_model: LLM model to use.

    Returns:
        Initial facts about the profile and a session ID for this conversation.
    """
    try:
        # Change LLM model if needed
        if selected_model != config.LLM_MODEL_ID:
            change_llm_model(selected_model)

        # Use a default URL for mock data if none provided
        if use_mock and not linkedin_url:
            linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

        # Extract profile data
        profile_data = extract_linkedin_profile(linkedin_url, api_key if not use_mock else None, mock=use_mock)

        if not profile_data:
            return "Failed to retrieve profile data. Please check the URL or API key.", None

        # Split data into nodes
        nodes = split_profile_data(profile_data)

        if not nodes:
            return "Failed to process profile data into nodes.", None

        # Create vector database
        index = create_vector_database(nodes)

        if not index:
            return "Failed to create vector database.", None

        # Verify embeddings
        if not verify_embeddings(index):
            logger.warning("Some embeddings may be missing or invalid")

        # Generate initial facts
        facts = generate_initial_facts(index)

        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # Store the index for this session
        active_indices[session_id] = index

        # Return the facts and session ID
        return (
            f"Profile processed successfully!\n\nHere are 3 interesting facts about this person:\n\n{facts}",
            session_id,
        )

    except Exception as e:
        logger.error(f"Error in process_profile: {e}")
        return f"Error: {str(e)}", None
```

### **How This Function Works:**

-   **Model Selection:** It changes the LLM model if the user selects a different one

-   **Data Extraction:** It extracts LinkedIn profile data, using mock data if specified

-   **Data Processing:** It splits the profile into nodes and creates a vector index

-   **Facts Generation:** It generates interesting facts about the profile

-   **Session Management:** It creates a unique session ID and stores the vector index for later use

-   **Error Handling:** It catches any exceptions and returns user-friendly error messages

## Step 3: Implement the Chat Function

Next, let's implement the function that handles chatting with the profile. Add the code below to `app.py`.

```python
def chat_with_profile(session_id, user_query, chat_history):
    """Chat with a processed LinkedIn profile.

    Args:
        session_id: Session ID for this conversation.
        user_query: User's question.
        chat_history: Chat history, as a list of {"role", "content"} messages.

    Returns:
        Updated chat history.
    """
    if not session_id:
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": "No profile loaded. Please process a LinkedIn profile first."},
        ]

    if session_id not in active_indices:
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": "Session expired. Please process the LinkedIn profile again."},
        ]

    if not user_query.strip():
        return chat_history

    try:
        # Get the index for this session
        index = active_indices[session_id]

        # Answer the user's query
        response = answer_user_query(index, user_query)

        # Update chat history
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": str(response)},
        ]

    except Exception as e:
        logger.error(f"Error in chat_with_profile: {e}")
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": f"Error: {str(e)}"},
        ]
```

### **How This Function Works:**

-   **Session Validation**: It checks if a profile has been processed and if the session is still active

-   **Query Processing**: It retrieves the vector index for the session and uses it to answer the query

-   **Chat History:** It appends the user's question and the bot's response to the chat history. Gradio 6's `Chatbot` stores the history as a list of messages, each a dictionary with a `role` (`"user"` or `"assistant"`) and a `content`; the `[question, answer]` pairs of older Gradio versions are no longer accepted

-   **Error Handling**: It catches any exceptions and adds error messages to the chat history

## Step 4: Create the Gradio Interface

Finally, let's create the Gradio interface. Add the code below to `app.py`.

```python
def create_gradio_interface():
    """Create the Gradio interface for the Icebreaker Bot."""
    # Define available LLM models (any model pulled into Ollama can be added here)
    available_models = [
        "qwen3-8b-rag",
    ]

    with gr.Blocks(title="LinkedIn Icebreaker Bot", analytics_enabled=False) as demo:
        gr.Markdown("# LinkedIn Icebreaker Bot")
        gr.Markdown("Generate personalized icebreakers and chat about LinkedIn profiles")

        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/username/"
                    )
                    api_key = gr.Textbox(
                        label="ProxyCurl API Key (Leave empty to use mock data)",
                        placeholder="Your ProxyCurl API Key",
                        type="password",
                        value=config.PROXYCURL_API_KEY,
                    )
                    use_mock = gr.Checkbox(label="Use Mock Data", value=True)
                    model_dropdown = gr.Dropdown(
                        choices=available_models, label="Select LLM Model", value=config.LLM_MODEL_ID
                    )
                    process_btn = gr.Button("Process Profile")

                with gr.Column():
                    result_text = gr.Textbox(label="Initial Facts", lines=10)
                    session_id = gr.Textbox(label="Session ID", visible=False)

            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock, model_dropdown],
                outputs=[result_text, session_id],
            )

        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile")

            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the profile", placeholder="What is this person's current job title?"
            )

            chat_btn = gr.Button("Send")

            chat_btn.click(fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot])

            chat_input.submit(fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot])

    return demo
```

### **How This Function Works:**

-   **Interface Structure:** It creates a tabbed interface with separate tabs for processing profiles and chatting

-   **Profile Processing Tab:** It provides fields for the LinkedIn URL, API key, mock data option, and model selection. The model list holds the one local model; add the name of any other model you pull into Ollama

-   **Chat Tab:** It provides a chat interface for asking questions about the profile

-   **Event Handlers:** It connects UI elements to our processing and chat functions. The hidden `Session ID` textbox carries the session ID from the first tab to the chat

-   **No analytics:** `analytics_enabled=False` stops Gradio from sending usage statistics

## Step 5: Launch the Interface

Lastly, let's add the code to launch the interface:

```python
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="127.0.0.1", server_port=5000)
```

This code creates and launches the Gradio interface when the script is run directly. `127.0.0.1` makes it reachable only from your own machine; use `0.0.0.0` instead if you want other devices on your network to reach it.

<details>
<summary>The full <code>app.py</code></summary>

```python
"""Gradio web interface for the Icebreaker Bot."""

import logging
import sys
import uuid

import gradio as gr

import config
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import create_vector_database, split_profile_data, verify_embeddings
from modules.llm_interface import change_llm_model
from modules.query_engine import answer_user_query, generate_initial_facts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

# Dictionary to store active conversations
active_indices = {}


def process_profile(linkedin_url, api_key, use_mock, selected_model):
    """Process a LinkedIn profile and generate initial facts.

    Args:
        linkedin_url: LinkedIn profile URL to process.
        api_key: ProxyCurl API key.
        use_mock: Whether to use mock data.
        selected_model: LLM model to use.

    Returns:
        Initial facts about the profile and a session ID for this conversation.
    """
    try:
        # Change LLM model if needed
        if selected_model != config.LLM_MODEL_ID:
            change_llm_model(selected_model)

        # Use a default URL for mock data if none provided
        if use_mock and not linkedin_url:
            linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

        # Extract profile data
        profile_data = extract_linkedin_profile(linkedin_url, api_key if not use_mock else None, mock=use_mock)

        if not profile_data:
            return "Failed to retrieve profile data. Please check the URL or API key.", None

        # Split data into nodes
        nodes = split_profile_data(profile_data)

        if not nodes:
            return "Failed to process profile data into nodes.", None

        # Create vector database
        index = create_vector_database(nodes)

        if not index:
            return "Failed to create vector database.", None

        # Verify embeddings
        if not verify_embeddings(index):
            logger.warning("Some embeddings may be missing or invalid")

        # Generate initial facts
        facts = generate_initial_facts(index)

        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # Store the index for this session
        active_indices[session_id] = index

        # Return the facts and session ID
        return (
            f"Profile processed successfully!\n\nHere are 3 interesting facts about this person:\n\n{facts}",
            session_id,
        )

    except Exception as e:
        logger.error(f"Error in process_profile: {e}")
        return f"Error: {str(e)}", None


def chat_with_profile(session_id, user_query, chat_history):
    """Chat with a processed LinkedIn profile.

    Args:
        session_id: Session ID for this conversation.
        user_query: User's question.
        chat_history: Chat history, as a list of {"role", "content"} messages.

    Returns:
        Updated chat history.
    """
    if not session_id:
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": "No profile loaded. Please process a LinkedIn profile first."},
        ]

    if session_id not in active_indices:
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": "Session expired. Please process the LinkedIn profile again."},
        ]

    if not user_query.strip():
        return chat_history

    try:
        # Get the index for this session
        index = active_indices[session_id]

        # Answer the user's query
        response = answer_user_query(index, user_query)

        # Update chat history
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": str(response)},
        ]

    except Exception as e:
        logger.error(f"Error in chat_with_profile: {e}")
        return chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": f"Error: {str(e)}"},
        ]


def create_gradio_interface():
    """Create the Gradio interface for the Icebreaker Bot."""
    # Define available LLM models (any model pulled into Ollama can be added here)
    available_models = [
        "qwen3-8b-rag",
    ]

    with gr.Blocks(title="LinkedIn Icebreaker Bot", analytics_enabled=False) as demo:
        gr.Markdown("# LinkedIn Icebreaker Bot")
        gr.Markdown("Generate personalized icebreakers and chat about LinkedIn profiles")

        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/username/"
                    )
                    api_key = gr.Textbox(
                        label="ProxyCurl API Key (Leave empty to use mock data)",
                        placeholder="Your ProxyCurl API Key",
                        type="password",
                        value=config.PROXYCURL_API_KEY,
                    )
                    use_mock = gr.Checkbox(label="Use Mock Data", value=True)
                    model_dropdown = gr.Dropdown(
                        choices=available_models, label="Select LLM Model", value=config.LLM_MODEL_ID
                    )
                    process_btn = gr.Button("Process Profile")

                with gr.Column():
                    result_text = gr.Textbox(label="Initial Facts", lines=10)
                    session_id = gr.Textbox(label="Session ID", visible=False)

            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock, model_dropdown],
                outputs=[result_text, session_id],
            )

        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile")

            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the profile", placeholder="What is this person's current job title?"
            )

            chat_btn = gr.Button("Send")

            chat_btn.click(fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot])

            chat_input.submit(fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot])

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="127.0.0.1", server_port=5000)
```

</details>

## Launching the application

Make sure Ollama is running, then run the Gradio app:

```bash
make app   # python app.py
```

After it starts, the terminal shows:

```
* Running on local URL:  http://127.0.0.1:5000
```

Open <http://127.0.0.1:5000> in your browser. On the **Process LinkedIn Profile** tab, leave **Use Mock Data** checked and click **Process Profile**. After the facts appear, switch to the **Chat** tab and ask a question.

To stop execution of `app.py`, hit `Ctrl+C` in the terminal.

## Troubleshooting

| Symptom                                                         | Fix                                                                                        |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `Connection refused` / `Failed to connect to Ollama` in the log | Ollama is not running: `make up`                                                           |
| `make up`: `port is already allocated` (11434)                  | Another Ollama is running, e.g. the module 2 lesson's: `docker compose down` in its folder |
| `Created 0 nodes`, and `NLTK will not read data` warnings       | The venv is group-writable: `chmod -R go-w .venv` (`make install` does it)                 |
| `model "qwen3-8b-rag" not found`                                | Redo setup step 3 (also needed after `docker compose down -v`)                             |
| Facts or answers contain reasoning text or a `</think>`         | `LLM_MODEL_ID` points at the raw `hf.co/...` model; use `qwen3-8b-rag`                     |
| `ModuleNotFoundError: No module named 'config'` or `'modules'`  | Run the scripts from inside this project folder                                            |
| `ModuleNotFoundError: No module named 'llama_index'`            | Use the project's venv: `make install`, then `make cli` / `make app` or `source .venv/bin/activate` |
| `Failed to retrieve profile data`                               | The mock profile could not be downloaded: check your internet connection                   |
| `ReadTimeout` on the first request                              | The model is still loading: try again, or raise `REQUEST_TIMEOUT` in `config.py`           |
| Very slow answers; `docker compose exec ollama ollama ps` shows `CPU` | The container can't see the GPU: run `docker run --rm --gpus all ubuntu nvidia-smi`  |
| Out of GPU memory                                               | Lower `CONTEXT_WINDOW`, or use the `Q4_K_M` quantization (5 GB): change it in the Makefile's `LLM_BASE` and on the `FROM` line of the `Modelfile`, then `make models` |
| `Cannot find empty port` / port 5000 in use                     | Another app is using port 5000: stop it, or change `server_port` in `app.py`               |

## Conclusion and next steps

### Conclusion

Congratulations on completing the AI Icebreaker Bot project! You've successfully built a powerful application that leverages Retrieval-Augmented Generation (RAG) and Large Language Models to transform networking and professional connections. By integrating a local Qwen3 model served by Ollama with the LlamaIndex framework, you've created a tool that can analyze LinkedIn profiles and generate personalized conversation starters, without sending any data to a cloud service.

This project has taken you through the complete RAG workflow — from data extraction and processing to indexing, retrieval, and generation. You've learned how to split complex JSON data into manageable chunks, create vector embeddings, build a vector index, and construct effective prompts for various tasks. With both a command-line interface and a Gradio web application, you've made your innovative tool accessible to users of different technical backgrounds.

As a final step, feel free to:

-   Test the bot with different LinkedIn profiles to see how it adapts to various career paths

-   Pull other models into Ollama and compare their performance on icebreaker generation, or try different configuration settings

-   Refine the prompts to generate even more engaging and personalized conversation starters

-   Store the index (for example in a Chroma service, like the one in the module 2 lesson's `compose.yaml`) so a profile does not have to be embedded again on every run

## Author(s)

Original lab: [Hailey Quach](https://www.haileyq.com/)

### Other Contributor(s)

Wojciech "Victor" Fulmyk, Data Scientist at IBM and PhD candidate at the University of Calgary (original lab).

This version replaces IBM watsonx.ai with a local Qwen3 served by Ollama.
