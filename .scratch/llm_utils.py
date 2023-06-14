import time
from functools import wraps

from ipdb import set_trace
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

collection_name = "tesla_earnings_call_transcripts"


def get_all_documents_with_payload(client: QdrantClient, *, year: int, quarter: int):
    documents = []

    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="meta.year", match=models.MatchValue(value=year)
                    ),
                    models.FieldCondition(
                        key="meta.quarter", match=models.MatchValue(value=quarter)
                    ),
                ]
            ),
            limit=10,
            offset=offset,
        )

        batch_documents = []
        for rec in records:
            payload = rec.payload
            if payload is None:
                raise Exception("payload is None")
            batch_document = Document(page_content=payload["text"])
            batch_documents.append(batch_document)

        documents.extend(batch_documents)

        if offset == None:
            break

    return documents


class Timer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time} seconds")
