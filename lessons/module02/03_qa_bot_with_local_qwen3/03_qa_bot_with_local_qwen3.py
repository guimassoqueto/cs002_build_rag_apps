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

from app.config.envs import CHROMA_HOST, CHROMA_PORT, OLLAMA_URL, QWEN3_EMBEDDING_MODEL, QWEN3_RAG_MODEL


# LLM: Qwen3-8B served by Ollama; reasoning=False skips thinking, sampling follows Qwen's non-thinking advice.
def get_llm():
    llm = ChatOllama(
        model=QWEN3_RAG_MODEL,
        base_url=OLLAMA_URL,
        reasoning=False,
        temperature=0.5,
        top_p=0.8,
        top_k=20,
        num_ctx=8192,
        num_predict=512,
    )
    return llm


# Document loader: Markdown is plain text, so each file becomes one Document tagged with its file name.
def document_loader(files):
    loaded_documents = [
        Document(page_content=Path(file).read_text(encoding="utf-8"), metadata={"source": Path(file).name})
        for file in files
    ]
    return loaded_documents


# Text splitter: cut at #/##/### headings first, then split long sections on Markdown boundaries.
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


# Vector db: the Chroma server in Docker; reset_collection() keeps earlier uploads out of the answers.
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


# Embedding model: Qwen3-Embedding-0.6B on the same Ollama server, used for both the chunks and the question.
def local_embedding():
    embedding = OllamaEmbeddings(
        model=QWEN3_EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )
    return embedding


# Retriever: load, split and store the files, then return the 4 chunks closest to the question.
def retriever(files):
    splits = document_loader(files)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever


# Prompt: the retrieved chunks go into the system message, and the model must stay within them.
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


# QA chain (LCEL): retrieve and "stuff" the chunks as context, fill the prompt, ask the LLM, keep the text.
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


# Gradio interface: upload one or more .md files, ask a question, read the answer; no analytics calls.
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
    description="Upload Markdown documents and ask any question. The chatbot answers using only the provided "
    "documents, with a model running on your machine.",
)

# launch() serves the bot on http://127.0.0.1:7860 only; no share link, so nothing leaves the machine.
rag_application.launch(server_name="127.0.0.1", server_port=7860)
