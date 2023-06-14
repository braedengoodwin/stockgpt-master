from typing import Dict

TESLA_CIK = "0001318605"


class EdgarClient:
    def __init__(self):
        ...

    async def _make_request(self, *, endpoint: str, params: Dict = {}):
        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = FMP_API_KEY

        url = f"https://financialmodelingprep.com{endpoint}"

        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params)

        return res
