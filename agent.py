from dataclasses import dataclass
from datetime import date

import openai
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import build_llm
from prompts import REACT_PROMPT
from tools import TOOLS

MAX_ITERATIONS = 8
MAX_HISTORY_MESSAGES = 20
PARSING_ERROR_MESSAGE = (
    "Formato inválido. Responda com Thought seguido de Action e Action Input, "
    "ou com Thought seguido de Final Answer."
)
ITERATION_LIMIT_MESSAGE = (
    "Não consegui concluir o raciocínio dentro do limite de etapas. "
    "Tente reformular a pergunta com mais detalhes."
)


class AgentError(Exception):
    """Erro com mensagem amigável para ser exibida ao estudante."""


@dataclass
class ReasoningStep:
    thought: str
    tool: str
    tool_input: str
    observation: str


@dataclass
class AgentReply:
    answer: str
    steps: list[ReasoningStep]


class StudyMentorAgent:
    """Agente ReAct que combina o LLM, as ferramentas de estudo e a memória de cada sessão."""

    def __init__(self, llm: BaseChatModel | None = None):
        prompt = REACT_PROMPT.partial(current_date=date.today().strftime("%d/%m/%Y"))
        react_agent = create_react_agent(llm or build_llm(), TOOLS, prompt)
        self._executor = AgentExecutor(
            agent=react_agent,
            tools=TOOLS,
            max_iterations=MAX_ITERATIONS,
            handle_parsing_errors=PARSING_ERROR_MESSAGE,
            return_intermediate_steps=True,
        )
        self._histories: dict[str, list[BaseMessage]] = {}

    def get_history(self, session_id: str) -> list[BaseMessage]:
        return self._histories.setdefault(session_id, [])

    def clear_history(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def ask(self, question: str, session_id: str) -> AgentReply:
        """Executa o ciclo raciocinar, agir e observar e devolve a resposta com as etapas percorridas."""
        question = question.strip()
        if not question:
            raise AgentError("Digite uma pergunta antes de enviar.")
        history = self.get_history(session_id)
        try:
            result = self._executor.invoke({"input": question, "chat_history": list(history)})
        except openai.AuthenticationError as exc:
            raise AgentError("Chave da OpenAI inválida. Verifique OPENAI_API_KEY no arquivo .env.") from exc
        except openai.RateLimitError as exc:
            raise AgentError("Limite de uso da OpenAI atingido. Aguarde alguns instantes e tente novamente.") from exc
        except openai.APIConnectionError as exc:
            raise AgentError("Não foi possível falar com a OpenAI. Verifique a conexão e tente novamente.") from exc
        except openai.APIError as exc:
            raise AgentError(f"Erro na API da OpenAI: {exc.message}") from exc

        answer = result["output"]
        if answer.startswith("Agent stopped"):
            answer = ITERATION_LIMIT_MESSAGE
        self._remember(history, question, answer)
        steps = [self._to_step(action, observation) for action, observation in result["intermediate_steps"]]
        return AgentReply(answer=answer, steps=steps)

    @staticmethod
    def _remember(history: list[BaseMessage], question: str, answer: str) -> None:
        """Guarda a troca no histórico e mantém só as mensagens mais recentes para limitar o prompt."""
        history.extend([HumanMessage(question), AIMessage(answer)])
        del history[:-MAX_HISTORY_MESSAGES]

    @staticmethod
    def _to_step(action, observation) -> ReasoningStep:
        thought = action.log.split("Action:")[0].split("Final Answer:")[0]
        thought = thought.strip().removeprefix("Thought:").strip()
        tool_name = "formato inválido" if action.tool == "_Exception" else action.tool
        return ReasoningStep(
            thought=thought,
            tool=tool_name,
            tool_input=str(action.tool_input),
            observation=str(observation),
        )
