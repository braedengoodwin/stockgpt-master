import os

from worker.clients import FinancialModelingPrepClient

fmp_client = FinancialModelingPrepClient(api_key="ab13f23f4ec7dbddfe57706d4a317d89")

balance_sheet = await fmp_client.get_financial_statement(
    "TSLA", statement="balance_sheet", period="annual", limit=1
)

cash_flow = await fmp_client.get_financial_statement(
    "TSLA", statement="cash_flow", period="annual", limit=1
)

income = await fmp_client.get_financial_statement(
    "TSLA", statement="income", period="annual", limit=1
)


def read_earnings_transcripts(symbol_name: str):
    transcripts = []

    directory = f"data/finance/{symbol_name}/earnings_transcripts"

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
