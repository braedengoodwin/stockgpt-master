import json
import os
from datetime import date
from typing import Dict, Literal

import httpx


class FinancialModelingPrepClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def _make_request(self, *, endpoint: str, params: Dict = {}):
        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = self.api_key

        url = f"https://financialmodelingprep.com{endpoint}"

        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params)

        return res

    async def get_financial_statement(
        self,
        symbol: str,
        *,
        statement: Literal["income", "balance_sheet", "cash_flow"],
        period: Literal["annual", "quarter"] = "annual",
        limit: int | None = None,
    ):
        endpoint = f"/api/v3/{statement.replace('_', '-')}-statement/{symbol}"
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

    async def get_stock_quote(self, symbol: str):
        endpoint = f"/api/v3/quote/{symbol}"
        res = await self._make_request(endpoint=endpoint)
        data = json.loads(res.content.decode())

        return data

    async def get_stock_quote_short(self, symbol: str):
        endpoint = f"/api/v3/quote-short/{symbol}"
        res = await self._make_request(endpoint=endpoint)
        data = json.loads(res.content.decode())

        return data

    async def get_historical_stock_price(
        self, symbol: str, *, from_date: date, to_date: date
    ):
        endpoint = f"/api/v3/historical-price-full/{symbol}"
        params = {"to": to_date, "from": from_date}
        res = await self._make_request(endpoint=endpoint, params=params)

        data = json.loads(res.content.decode())

        return data
