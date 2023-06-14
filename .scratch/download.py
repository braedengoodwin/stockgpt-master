import json
import os
from enum import Enum
from typing import Dict, Literal

import httpx
from ipdb import set_trace
from tqdm import tqdm

FMP_API_KEY = "ab13f23f4ec7dbddfe57706d4a317d89"


def get_xlsx():
    res = httpx.get(
        "https://financialmodelingprep.com/api/v4/financial-reports-xlsx",
        params={"apikey": FMP_API_KEY, "symbol": "AAPL", "year": 2022, "period": "Q1"},
    )
    with open("data.xlsx", "wb") as f:
        f.write(res.content)


def get_earnings_call_transcript(*, symbol: str, year: int, quarter: int):
    res = httpx.get(
        f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}",
        params={"apikey": FMP_API_KEY, "year": year, "quarter": quarter},
    )
    data = json.loads(res.content.decode())

    transcript = data[0]["content"]

    return transcript


def download_earnings_call_transcripts(*, symbol: str):
    available_transcripts_res = httpx.get(
        "https://financialmodelingprep.com/api/v4/earning_call_transcript",
        params={"apikey": FMP_API_KEY, "symbol": symbol},
    )
    available_transcripts_data = json.loads(available_transcripts_res.content.decode())

    transcript_filepath = f"data/{symbol}/earnings_transcripts"
    if not os.path.exists(transcript_filepath):
        os.makedirs(transcript_filepath)

    for quarter, year, _ in tqdm(available_transcripts_data):
        transcript = get_earnings_call_transcript(
            symbol=symbol, year=year, quarter=quarter
        )
        transcript_filename = f"{transcript_filepath}/q{quarter}_{year}.txt"
        with open(transcript_filename, "w") as f:
            f.write(transcript)


class FinancialStatement(Enum):
    INCOME = "income"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class FinancialModelingPrepClient:
    def __init__(self):
        ...

    async def _make_request(self, *, endpoint: str, params: Dict = {}):
        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = FMP_API_KEY

        url = f"https://financialmodelingprep.com{endpoint}"

        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params)

        return res

    async def get_financial_statement(
        self,
        symbol: str,
        *,
        statement: FinancialStatement,
        period: Literal["annual", "quarter"] = "annual",
        limit: int | None = None,
    ):
        endpoint = f"/api/v3/{statement.value.replace('_', '-')}-statement/{symbol}"
        params = {"period": period, "limit": limit}
        res = await self._make_request(endpoint=endpoint, params=params)

        data = json.loads(res.content.decode())

        return data

    async def get_form_10k(self, symbol: str, *, year: int):
        endpoint = "/api/v4/financial-reports-json"
        params = {"symbol": symbol, "year": year, "period": "FY"}
        res = await self._make_request(endpoint=endpoint, params=params)

        data = json.loads(res.content.decode())

        return data

    async def get_quartery_reports(self, symbol: str, *, year: int, quarter: int):
        endpoint = "/api/v4/financial-reports-json"
        params = {"symbol": symbol, "year": year, "period": f"Q{quarter}"}
        res = await self._make_request(endpoint=endpoint, params=params)

        data = json.loads(res.content.decode())

        return data

    async def download_form_10q(
        self,
        *,
        symbol: str,
        year: int,
        quarter: Literal[1, 2, 3, 4],
        filetype: Literal["json", "xlsx"],
    ):
        endpoint = f"/api/v4/financial-reports-{filetype}"
        params = {"symbol": symbol, "year": year, "period": f"Q{quarter}"}
        res = await self._make_request(endpoint=endpoint, params=params)

        filepath = f"data/{symbol}/10q"
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filename = f"{filepath}/{year}.{filetype}"
        if filetype == "json":
            data = res.content.decode()
            with open(filename, "w") as f:
                f.write(data)
        elif filetype == "xlsx":
            data = res.content
            with open(filename, "wb") as f:
                f.write(data)

        print("File saved at:", filename)

    async def get_financial_report_dates(self, symbol: str):
        endpoint = "/api/v4/financial-reports-dates"
        params = {"symbol": symbol}
        res = await self._make_request(endpoint=endpoint, params=params)

        data = json.loads(res.content.decode())

        return data


def save_10k_to_json(*, symbol: str, year: int, data: Dict):
    filepath = f"data/{symbol}/10k"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = f"{filepath}/{year}.json"
    with open(filename, "w") as f:
        f.write(json.dumps(data))


async def main():
    #

    from download import FinancialModelingPrepClient

    client = FinancialModelingPrepClient()

    await client.download_form_10q(symbol="AAPL", year=2020, quarter=1, filetype="xlsx")

    fr_dates = await client.get_financial_report_dates("ED")

    res = await client.get_financial_statement(
        "AAPL", statement=FinancialStatement.CASH_FLOW, period="quarter", limit=2
    )

    form_10k = await client.get_form_10k("AAPL", year=2020)

    save_10k_to_json(symbol="AAPL", year=2020, data=form_10k)

    form_10q = await client.get_quartery_reports("AAPL", year=2020, quarter=1)


def download_10k(*, symbol: str, year: int, filetype: Literal["json", "xlsx"]):
    filepath = f"data/{symbol}/10k"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    res = httpx.get(
        f"https://financialmodelingprep.com/api/v4/financial-reports-{filetype}",
        params={"apikey": FMP_API_KEY, "symbol": symbol, "year": year, "period": "FY"},
    )

    filename = f"{filepath}/{year}.{filetype}"
    if filetype == "json":
        data = res.content.decode()
        with open(filename, "w") as f:
            f.write(data)
    elif filetype == "xlsx":
        data = res.content
        with open(filename, "wb") as f:
            f.write(data)
