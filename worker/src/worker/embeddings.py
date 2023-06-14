import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .utils import chunked_list


def read_earnings_transcripts(symbol_name: str):
    transcripts = []

    directory = f"data/{symbol_name}/earnings_transcripts"

    files = os.listdir(directory)
    for file_name in files:
        with open(os.path.join(directory, file_name), "r") as file:
            content = file.read()

        quarter, year = file_name.split("_")
        quarter = int(quarter[1])
        year = int(year.split(".")[0])

        transcript = {
            "content": content,
            "quarter": quarter,
            "year": year,
            "symbol": symbol_name,
        }

        transcripts.append(transcript)

    return transcripts


def build_documents(*, data: List[Dict], text_splitter: RecursiveCharacterTextSplitter):
    documents = []
    for d in data:
        document = Document(
            page_content=d["content"],
            metadata={k: v for k, v in d.items() if k != "content"},
        )
        documents.append(document)
    split_documents = text_splitter.split_documents(documents)

    return split_documents


# 2. EMBED DOCUMENTS


def build_embeddings(
    *,
    documents: List[Document],
    embedding_func: Callable,
    batch_size: int = 64,
    saved_embeddings_filepath: str,
):
    embeddings = []
    for batch in tqdm(chunked_list(documents, batch_size)):
        batch_content = [d.page_content for d in batch]
        batch_embeddings = embedding_func(batch_content)
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings)

    np.save(saved_embeddings_filepath, embeddings, allow_pickle=False)

    return embeddings


def load_embeddings(saved_embeddings_filepath: str):
    return np.load(saved_embeddings_filepath)
