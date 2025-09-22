from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings and secrets using Pydantic.
    It automatically reads environment variables from a .env file.
    """
    OPENAI_API_KEY: str
    TAVILY_API_KEY: str

    # Define the location of the .env file.
    # The path is relative to the project's root directory.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

# Create a singleton instance of settings
settings = Settings()

