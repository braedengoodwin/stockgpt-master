import asyncio
from datetime import date

from langchain.tools import Tool
from pydantic import BaseModel, Field

from worker.clients import FinancialModelingPrepClient
from worker.settings import settings

fmp_client = FinancialModelingPrepClient(api_key=settings.FMP_API_KEY)


class ToolInputSchema(BaseModel):
    symbol: str = Field(
        ..., description="A stock ticker symbole. Ex: 'AAPL', 'TSLA', 'AMZN'"
    )


def _get_current_stock_quote(symbol: str):
    fmp_client = FinancialModelingPrepClient(api_key=settings.FMP_API_KEY)
    res = asyncio.run(fmp_client.get_stock_quote(symbol))

    return res


get_current_stock_quote = Tool.from_function(
    func=_get_current_stock_quote,
    name="get_current_stock_quote",
    description="Useful for when you need to get the current stock quote",
    args_schema=ToolInputSchema,
    return_direct=True,
)
