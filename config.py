import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HTTP_TIMEOUT_SECONDS = 15
HTTP_HEADERS = {"User-Agent": "MentorDeEstudos/1.0 (projeto academico)"}


class ConfigurationError(Exception):
    """Erro de configuração do ambiente, como a falta da chave da OpenAI."""


def build_llm() -> ChatOpenAI:
    """Cria o modelo da OpenAI usado pelo agente, validando a chave de acesso."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ConfigurationError("Defina a variável OPENAI_API_KEY no arquivo .env antes de iniciar o agente.")
    return ChatOpenAI(model=MODEL_NAME, temperature=0, timeout=60, max_retries=2)
