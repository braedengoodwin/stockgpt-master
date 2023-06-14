import asyncio
from datetime import date

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from worker.clients import FinancialModelingPrepClient
from worker.settings import settings

fmp_client = FinancialModelingPrepClient(api_key=settings.FMP_API_KEY)


class ToolInputSchema(BaseModel):
    symbol: str = Field(
        ..., description="A stock ticker symbole. Ex: 'AAPL', 'TSLA', 'AMZN'"
    )
    from_date: date = Field(
        ..., description="The starting date to get historical price data from"
    )
    to_date: date = Field(
        ..., description="The ending date to get historical price data from"
    )


def _get_historical_stock_price(symbol: str, from_date: date, to_date: date):
    fmp_client = FinancialModelingPrepClient(api_key=settings.FMP_API_KEY)
    res = asyncio.run(
        fmp_client.get_historical_stock_price(
            symbol, from_date=from_date, to_date=to_date
        )
    )

    return res


get_historical_stock_price = StructuredTool.from_function(
    func=_get_historical_stock_price,
    name="get_historical_stock_price",
    description="Useful for when you need the price history of a stock",
    args_schema=ToolInputSchema,
    return_direct=True,
)
