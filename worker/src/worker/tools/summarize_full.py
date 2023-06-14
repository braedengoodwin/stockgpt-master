import os
from functools import partial

import torch
from bertopic import BERTopic
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from worker.utils import hf_len

ROOT_DIR = "/home/waydegg/ghq/github.com/waydegg/stockgpt/worker"


def summarize_earnings_call_full(
    *, symbol: str, year: int, quarter: int, verbose: bool = False
):
    # Find earnings call file
    symbol_dir = f"{ROOT_DIR}/data/finance/{symbol.upper()}/earnings_transcripts"
    if not os.path.exists(symbol_dir):
        return f"No earnings calls for symbol {symbol}"
    symbol_filename = f"q{quarter}_{year}.txt"
    symbol_filepath = f"{symbol_dir}/{symbol_filename}"
    if not os.path.exists(symbol_filepath):
        return f"No earnings call for symbol {symbol} on Q{quarter} in {year}"
    with open(symbol_filepath, "r") as f:
        earnigns_call_document = Document(
            page_content=f.read(),
            metadata={"symbol": symbol, "year": year, "quarter": quarter},
        )

    # Load embedding model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()  # type: ignore
        else "cpu"
    )
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    # Call summarize chain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=100,
        length_function=partial(hf_len, model.tokenizer),
    )
    split_earnigns_call_documents = text_splitter.split_documents(
        [earnigns_call_document]
    )
    chain = load_summarize_chain(
        llm=ChatOpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-3.5-turbo",
            client=None,
            temperature=0,
        ),
        chain_type="map_reduce",
        verbose=verbose,
    )
    res = chain.run(split_earnigns_call_documents)

    return res
