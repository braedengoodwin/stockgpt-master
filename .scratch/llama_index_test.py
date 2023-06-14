import logging
import os
import sys

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (LangchainEmbedding, LLMPredictor, ServiceContext,
                         SimpleDirectoryReader, VectorStoreIndex)
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.storage.storage_context import StorageContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

## Custom Embeddings

embedding_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

## Load data


def get_metadata(filename: str):
    symbol, form, filing_date = filename.split("_")
    metadata = {"symbol": symbol, "form": form, "filing_date": filing_date}
    return metadata


oct_2021 = SimpleDirectoryReader(
    input_files=["./data/TSLA/10q/TSLA_10Q_2021-10-25.pdf"],
).load_data()
apr_2022 = SimpleDirectoryReader(
    input_files=["./data/TSLA/10q/TSLA_10Q_2022-04-25.pdf"],
).load_data()
jul_2022 = SimpleDirectoryReader(
    input_files=["./data/TSLA/10q/TSLA_10Q_2022-07-25.pdf"],
).load_data()
oct_2022 = SimpleDirectoryReader(
    input_files=["./data/TSLA/10q/TSLA_10Q_2022-10-24.pdf"],
).load_data()
apr_2023 = SimpleDirectoryReader(
    input_files=["./data/TSLA/10q/TSLA_10Q_2023-04-24.pdf"],
).load_data()


vector_store = QdrantVectorStore(
    client=QdrantClient("http://localhost:6333"), collection_name="test1"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-3.5-turbo",
        client=None,
        temperature=0,
        # max_tokens=4096,
    )
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, embed_model=embedding_model
)

## Create indicies


def create_vector_store_index(document: str):
    documents = SimpleDirectoryReader(input_files=[document]).load_data()

    collection_name = document.split("/")[-1].split(".")[0]
    vector_store = QdrantVectorStore(
        client=QdrantClient("http://localhost:6333"), collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_store_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
    )

    return vector_store_index


oct_2021_index = create_vector_store_index("./data/TSLA/10q/TSLA_10Q_2021-10-25.pdf")
apr_2022_index = create_vector_store_index("./data/TSLA/10q/TSLA_10Q_2022-04-25.pdf")
jul_2022_index = create_vector_store_index("./data/TSLA/10q/TSLA_10Q_2022-07-25.pdf")
oct_2022_index = create_vector_store_index("./data/TSLA/10q/TSLA_10Q_2022-10-24.pdf")
apr_2023_index = create_vector_store_index("./data/TSLA/10q/TSLA_10Q_2023-04-24.pdf")

## Build query engines

oct_2021_engine = oct_2021_index.as_query_engine(similarity_top_k=3)
apr_2022_engine = apr_2022_index.as_query_engine(similarity_top_k=3)
jul_2022_engine = jul_2022_index.as_query_engine(similarity_top_k=3)
oct_2022_engine = oct_2022_index.as_query_engine(similarity_top_k=3)
apr_2023_engine = apr_2023_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=oct_2021_engine,
        metadata=ToolMetadata(
            name="september_2021",
            description="Provides information about Tesla quarterly financials ending September 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=apr_2022_engine,
        metadata=ToolMetadata(
            name="march_2022",
            description="Provides information about Tesla quarterly financials ending March 2022",
        ),
    ),
    QueryEngineTool(
        query_engine=jul_2022_engine,
        metadata=ToolMetadata(
            name="june_2022",
            description="Provides information about Tesla quarterly financials ending June 2022",
        ),
    ),
    QueryEngineTool(
        query_engine=oct_2022_engine,
        metadata=ToolMetadata(
            name="september_2022",
            description="Provides information about Tesla quarterly financials ending September 2022",
        ),
    ),
    QueryEngineTool(
        query_engine=apr_2023_engine,
        metadata=ToolMetadata(
            name="march_2023",
            description="Provides information about Tesla quarterly financials ending March 2023",
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

res = s_engine.query("Breakdown Tesla's Cash Flows from Operating Activities for 2023.")


## FMP data

from download import FinancialModelingPrepClient

fmp_client = FinancialModelingPrepClient()
