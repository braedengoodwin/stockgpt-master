import os
import time
from functools import partial

import torch
from langchain.agents import AgentType, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import StructuredTool
from langchain.vectorstores import Qdrant
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from worker.utils import build_embeddings, load_earnings_call_transcripts

# Data

## Form 10K


loader = UnstructuredPDFLoader("./data/finance/TSLA/10k/2022.pdf")
documents = loader.load()
document = documents[0]

## Earnings Call Transcripts


call_documents = load_earnings_call_transcripts("TSLA")

## Embeddings

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)
print("Device:", device)


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)


def hf_len(tokenizer, text):
    return len(tokenizer.tokenize(text))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=partial(hf_len, model.tokenizer),
)

### Form 10K

split_documents = text_splitter.split_documents([document])

embeddings = build_embeddings(
    documents=split_documents,
    embedding_func=model.encode,
    batch_size=256,
    saved_embeddings_filepath="./data/embeddings/tsla_10k.npy",
)

### Earnings Call Transcripts

split_call_documents = text_splitter.split_documents(call_documents)

call_embeddings = build_embeddings(
    documents=split_call_documents,
    embedding_func=model.encode,
    batch_size=512,
    saved_embeddings_filepath="./data/embeddings/tsla_earnings_calls.npy",
)

## Save embeddings to vector store

qdrant_client = QdrantClient("http://localhost:6333")

### Form 10K

collection_name = "tsla_10k"

qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
qdrant_client.upload_collection(
    collection_name=collection_name,
    vectors=[e.tolist() for e in embeddings],
    payload=[{"text": sd.page_content, "meta": sd.metadata} for sd in split_documents],
    ids=None,
    batch_size=256,
    parallel=1,
)

### Earnings Call Transcripts

collection_name = "tsla_earnings_call_transcripts"

qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
qdrant_client.upload_collection(
    collection_name=collection_name,
    vectors=[e.tolist() for e in call_embeddings],
    payload=[
        {"text": sd.page_content, "meta": sd.metadata} for sd in split_call_documents
    ],
    ids=None,
    batch_size=512,
    parallel=1,
)


## Langchain

### Setup

llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo",
    # model="gpt-3.5-turbo",
    client=None,
    temperature=0,
)

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    ),
    content_payload_key="text",
    metadata_payload_key="meta",
)

retriever = vectorstore.as_retriever(
    verbose=True,
    search_kwargs={"k": 10},
)

# docs = retreiver.get_relevant_documents("battey safety")

### Retrieval Q/A

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

qa.run("Where are Tesla's manufacturing facilities located?")

print(qa.run("Give me an itemized list of all of Tesla's revenues for 2022"))

print(qa.run("What was Tesla's total revenue for 2022?"))

print(qa.run("How much did Tesla's leasing revenue increase this past year?"))

print(qa.run("What was Tesla's total cost of revenues for 2022"))

print(qa.run("Tell me about Teslas energy generation and storage revenue"))

print(qa.run("Tell me about Tesla's deliveries this past year"))


### Summary

#### Summarize a single document

ROOT_DIR = "/home/waydegg/ghq/github.com/waydegg/stockgpt/worker"


def summarize_earnings_call(*, symbol: str, year: int, quarter: int):
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

    # Call summarize chain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=100,
        length_function=partial(hf_len, model.tokenizer),
    )
    split_earnigns_call_documents = text_splitter.split_documents(
        [earnigns_call_document]
    )
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)
    res = chain.run(split_earnigns_call_documents)

    return res


summarize_earnings_call(symbol="tsla", year=2020, quarter=3)


class ToolInputSchema(BaseModel):
    symbol: str = Field(..., description="A stock ticker symbol")
    year: int = Field(..., description="A calendar year")
    quarter: int = Field(..., description="A calendar quarter")


tool = StructuredTool.from_function(
    # func=summarize_earnings_call,
    func=summarize_earnings_call_2,
    name="summarize_earnings_call",
    description="Summarize a single earnings call transcript for a specific stock ticker at a specific year and quarter.",
    args_schema=ToolInputSchema,
)

tools = [tool]

test_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

test_agent.run("Summarize the earnings call transcript for TSLA in Q4 of 2018")


### Topic modeling summary (Tool)

from worker.tools.summarize import summarize_earnings_call
from worker.tools.summarize_full import summarize_earnings_call_full

res = summarize_earnings_call(symbol="tsla", year=2019, quarter=3, verbose=False)
print(res)

full_res = summarize_earnings_call_full(
    symbol="tsla", year=2019, quarter=3, verbose=False
)
print(full_res)

### Agent with some tools

import os
from datetime import date, datetime

from langchain.agents import AgentType, StructuredChatAgent, initialize_agent
from langchain.chat_models import ChatOpenAI

from worker.clients import FinancialModelingPrepClient
from worker.settings import settings
from worker.tools.get_current_stock_price import get_current_stock_price
from worker.tools.get_current_stock_quote import get_current_stock_quote
from worker.tools.get_historical_stock_price import get_historical_stock_price

fmp_client = FinancialModelingPrepClient(api_key=settings.FMP_API_KEY)

await fmp_client.get_historical_stock_price(
    "TSLA",
    from_date=date(year=2020, month=1, day=1),
    to_date=date(year=2020, month=2, day=1),
)

tools = [get_current_stock_price, get_current_stock_quote, get_historical_stock_price]

# prompt = StructuredChatAgent.create_prompt(
#     tools=tools,
#     suffix=f"The current UTC date is {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
# )

agent = initialize_agent(
    tools,
    llm=ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-3.5-turbo",
        client=None,
        temperature=0,
    ),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": f"The current UTC date is {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    },
)


agent.run("Get the stock price for TSLA")

agent.run("What's the latest quote for AAPL?")

agent.run("What is the price history for TSLA over the last 3 days?")

agent.run("What is the price history for TSLA over the last 3 days?")
