from pydantic import BaseSettings


class Settings(BaseSettings):
    FMP_API_KEY: str


settings = Settings()  # pyright: ignore
