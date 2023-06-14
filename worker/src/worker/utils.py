import os
from itertools import islice
from typing import Callable, List

import numpy as np
from langchain.schema import Document
from tqdm import tqdm


def hf_len(tokenizer, text):
    return len(tokenizer.tokenize(text))


def chunked_list(lst, chunk_size):
    """
    Example usage:

        my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        chunk_size = 3

        for chunk in chunked_list(my_list, chunk_size):
            print(chunk)
    """
    it = iter(lst)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


def build_embeddings(
    *,
    documents: List[Document],
    embedding_func: Callable,
    batch_size: int = 64,
    saved_embeddings_filepath: str | None = None,
):
    embeddings = []
    for batch in tqdm(chunked_list(documents, batch_size)):
        batch_content = [d.page_content for d in batch]
        batch_embeddings = embedding_func(batch_content)
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings)

    if saved_embeddings_filepath:
        np.save(saved_embeddings_filepath, embeddings, allow_pickle=False)

    return embeddings


def load_embeddings(saved_embeddings_filepath: str):
    return np.load(saved_embeddings_filepath)


def load_earnings_call_transcripts(symbol_name: str):
    documents = []

    directory = f"data/finance/{symbol_name}/earnings_transcripts"

    files = os.listdir(directory)
    for file_name in files:
        with open(os.path.join(directory, file_name), "r") as file:
            content = file.read()

        quarter, year = file_name.split("_")
        quarter = int(quarter[1])
        year = int(year.split(".")[0])

        document = Document(
            page_content=content,
            metadata={
                "symbol": symbol_name,
                "year": year,
                "quarter": quarter,
            },
        )
        documents.append(document)

    return documents
