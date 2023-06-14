import os
from functools import partial

import torch
from bertopic import BERTopic
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from worker.utils import build_embeddings, hf_len

ROOT_DIR = "/home/waydegg/ghq/github.com/waydegg/stockgpt/worker"


def summarize_earnings_call(
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
        document = Document(
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

    # Build embeddings
    print("Building embeddings...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=partial(hf_len, model.tokenizer),
    )
    split_document = text_splitter.split_documents([document])
    embeddings = build_embeddings(
        documents=split_document,
        embedding_func=model.encode,
        batch_size=512,
    )

    # Fit topic model
    print("Fitting topic model...")
    topic_model = BERTopic()
    topic_model.fit_transform(
        documents=[d.page_content for d in split_document], embeddings=embeddings
    )
    topic_df = topic_model.get_document_info([d.page_content for d in split_document])

    # Get top 3 topics groups
    topic_df = topic_model.get_topic_info()
    topic_df = topic_df[(topic_df.Topic >= 0) & (topic_df.Topic <= 2)]
    topic_groups = topic_df[["Topic", "Representative_Docs"]].values.tolist()

    # Build the prompt and input text
    prompt_topic_chunks = []
    for topic_num, topic_docs in topic_groups:
        topic_docs_text = "\n".join(topic_docs)
        prompt_topic_chunk = f"""
        TOPIC {topic_num}:
        
        {topic_docs_text}
        """
        prompt_topic_chunks.append(prompt_topic_chunk)
    prompt_input_text = "\n".join(prompt_topic_chunks)

    template = """
    Write a concise summary of the following snippets from an Earnings Transcript. 
    The following snippets come from a larger document and are grouped together by topic:  
    
    {text}

    CONCISE SUMMARY:"""
    prompt_template = PromptTemplate(template=template, input_variables=["text"])

    # Call summarize chain
    print("Calling summarize chain...")
    chain = load_summarize_chain(
        llm=ChatOpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-3.5-turbo",
            client=None,
            temperature=0,
        ),
        chain_type="stuff",
        verbose=verbose,
        prompt=prompt_template,
    )
    docs = [Document(page_content=prompt_input_text)]
    res = chain.run(docs)

    return res


class ToolInputSchema(BaseModel):
    symbol: str = Field(..., description="A stock ticker symbol")
    year: int = Field(..., description="A calendar year")
    quarter: int = Field(..., description="A calendar quarter")


summarize_earnings_call_tool = StructuredTool.from_function(
    func=summarize_earnings_call,
    name="summarize_earnings_call",
    description="Summarize a single earnings call transcript for a specific stock ticker at a specific year and quarter.",
    args_schema=ToolInputSchema,
)
