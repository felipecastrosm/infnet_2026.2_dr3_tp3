import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import tools
from agent import AgentError, StudyMentorAgent


class RecordingChatModel(FakeListChatModel):
    received_prompts: list = []

    def _record(self, messages):
        self.received_prompts.append("\n".join(str(message.content) for message in messages))

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self._record(messages)
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._record(messages)
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture(autouse=True)
def offline_holidays(monkeypatch):
    tools.fetch_national_holidays.cache_clear()
    monkeypatch.setattr(tools, "_fetch_json", lambda *args, **kwargs: [{"date": "2026-12-25", "name": "Natal"}])


def test_agent_reasons_acts_and_observes():
    llm = RecordingChatModel(
        received_prompts=[],
        responses=[
            "Preciso calcular os dias.\nAction: calcular_dias_de_estudo\nAction Input: 2026-12-21; 2026-12-27; domingo",
            "Já tenho os dados.\nFinal Answer: Você terá 5 dias de estudo.",
        ],
    )
    reply = StudyMentorAgent(llm=llm).ask("Quantos dias tenho até 27/12?", "sessao")

    assert reply.answer == "Você terá 5 dias de estudo."
    assert len(reply.steps) == 1
    assert reply.steps[0].thought == "Preciso calcular os dias."
    assert reply.steps[0].tool == "calcular_dias_de_estudo"
    assert "Dias disponíveis para estudo: 5" in reply.steps[0].observation
    assert "Observation: Período" in llm.received_prompts[1]


def test_agent_remembers_previous_messages():
    llm = RecordingChatModel(
        received_prompts=[],
        responses=[
            "Sem ferramentas.\nFinal Answer: Anotado, sua prova é em 20/12.",
            "Uso o histórico.\nFinal Answer: Sua prova é em 20/12.",
        ],
    )
    agent = StudyMentorAgent(llm=llm)
    agent.ask("Minha prova é em 20/12.", "sessao")
    agent.ask("Quando é minha prova?", "sessao")

    assert "Minha prova é em 20/12." in llm.received_prompts[1]
    assert len(agent.get_history("sessao")) == 4


def test_sessions_have_isolated_memory():
    llm = RecordingChatModel(
        received_prompts=[],
        responses=["Ok.\nFinal Answer: Anotado.", "Ok.\nFinal Answer: Não sei a data."],
    )
    agent = StudyMentorAgent(llm=llm)
    agent.ask("Minha prova é em 20/12.", "aluno_a")
    agent.ask("Quando é minha prova?", "aluno_b")

    assert "Minha prova é em 20/12." not in llm.received_prompts[1]


def test_clear_history_forgets_conversation():
    llm = RecordingChatModel(received_prompts=[], responses=["Ok.\nFinal Answer: Anotado."])
    agent = StudyMentorAgent(llm=llm)
    agent.ask("Estudo 3 horas por dia.", "sessao")
    agent.clear_history("sessao")

    assert agent.get_history("sessao") == []


def test_agent_recovers_from_invalid_format():
    llm = RecordingChatModel(
        received_prompts=[],
        responses=["resposta fora do formato", "Corrigindo.\nFinal Answer: Pronto."],
    )
    reply = StudyMentorAgent(llm=llm).ask("Oi", "sessao")

    assert reply.answer == "Pronto."
    assert reply.steps[0].tool == "formato inválido"


def test_agent_rejects_empty_question():
    agent = StudyMentorAgent(llm=RecordingChatModel(received_prompts=[], responses=["x"]))
    with pytest.raises(AgentError):
        agent.ask("   ", "sessao")
