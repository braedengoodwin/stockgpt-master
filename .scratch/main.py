import torch
from embeddings import (build_documents, build_embeddings, get_text_splitter,
                        load_embeddings, read_earnings_transcripts)
from sentence_transformers import SentenceTransformer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)

## Documents

data = read_earnings_transcripts("TSLA")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
text_splitter = get_text_splitter(model)
documents = build_documents(data=data, text_splitter=text_splitter)

## Embeddings

# embeddings = build_embeddings(
#     documents=documents,
#     embedding_func=model.encode,
#     saved_embeddings_filepath="./data/embeddings/tesla_earnings_call_transcripts.npy",
# )

embeddings = load_embeddings("./data/embeddings/tesla_earnings_call_transcripts.npy")

# 3. STORE EMBEDDED DOCUMENTS (VECTORS)

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

qdrant_client = QdrantClient("http://localhost:6333")

collection_name = "tesla_earnings_call_transcripts"

# qdrant_client.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )
#
# qdrant_client.upload_collection(
#     collection_name=collection_name,
#     vectors=[e.tolist() for e in embeddings],
#     payload=[{"text": d.page_content, "meta": d.metadata} for d in documents],
#     ids=None,
#     batch_size=256,
#     parallel=1,
# )

# 4. Do LLM stuff


import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Qdrant

llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo",
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

## Self-querying

from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(
        name="quater",
        description="The fiscal quarter the earnings call took place",
        type="integer",
    ),
    AttributeInfo(
        name="year", description="The year the earnings call took place", type="integer"
    ),
    AttributeInfo(
        name="symbol",
        description="The stock ticker symbol for the earnings call",
        type="string",
    ),
]
document_content_description = (
    "A call transcript of a pulbic company discussing their financial performance"
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True,
    search_kwargs={"k": 10},
)

query = "production for Q1 of 2020 for Tesla?"
docs = retriever.get_relevant_documents(query)
len(docs)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What was discussed in the earnings call for Q1 of 2020 for Tesla?"
qa.run(query)

## Base VesctorStore retriever

vs_retreiver = vectorstore.as_retriever(
    search_kwargs={"k": 10, "filter": {"year": 2021, "quarter": 1}}
)

query = "Tell me about production line issues for Q1 of 2020 for Tesla"
docs = vs_retreiver.get_relevant_documents(query)


## Summarize

from langchain.chains.summarize import load_summarize_chain
from llm_utils import Timer, get_all_documents_with_payload
from qdrant_client.http import models

docs = get_all_documents_with_payload(qdrant_client, year=2020, quarter=2)

chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)


with Timer():
    res = chain.run(docs[:15])
print(res)

# Llama Index

from llama_index import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.readers.qdrant import QdrantReader
from llama_index.vector_stores import QdrantVectorStore

reader = QdrantReader("http://localhost:6333")

documents = reader.load_data(collection_name=collection_name)

storage_context = StorageContext.from_defaults(
    vector_store=QdrantVectorStore(
        collection_name=collection_name, client=qdrant_client
    )
)

index = VectorStoreIndex(storage_context=storage_context)

documents = SimpleDirectoryReader("./data/TSLA/earnings_transcripts").load_data()

parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)


### Langchain JSON agent

import os

from download import FinancialModelingPrepClient
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chat_models import ChatOpenAI
from langchain.tools.json.tool import JsonSpec

fmp_client = FinancialModelingPrepClient()

form_10k = await fmp_client.get_form_10k("AAPL", year=2019)

json_spec = JsonSpec(dict_=form_10q, max_value_length=8000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-3.5-turbo",
        client=None,
        temperature=0,
    ),
    toolkit=json_toolkit,
    verbose=True,
)

json_agent_executor.run("How does Apple calculate Debt?")
