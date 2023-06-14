from functools import partial
from itertools import islice
from typing import Callable

import torch
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Data

loader = UnstructuredPDFLoader("./data/TSLA/10k/2022.pdf")
data = loader.load()

## Embeddings

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)


def hf_len(tokenizer, text):
    return len(tokenizer.tokenize(text))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=partial(hf_len, model.tokenizer),
)


def chunked_list(lst, chunk_size):
    it = iter(lst)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


def build_embeddings(
    *,
    document: Document,
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
