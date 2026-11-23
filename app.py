import uuid

import streamlit as st

from agent import AgentError, AgentReply, StudyMentorAgent
from config import MODEL_NAME, ConfigurationError

EXAMPLE_QUESTIONS = [
    "Tenho prova do ENEM em 2026-12-20 e estudo 3 horas por dia, menos aos domingos. Quantas horas terei?",
    "Divida essas horas entre Matemática, Redação e Biologia, sendo Matemática a mais difícil.",
    "Explique o que é estequiometria para eu revisar.",
]

st.set_page_config(page_title="Mentor de Estudos")


@st.cache_resource
def load_agent() -> StudyMentorAgent:
    """Cria o agente uma única vez e reaproveita entre as interações."""
    return StudyMentorAgent()


def render_steps(reply: AgentReply) -> None:
    """Mostra as etapas Thought, Action e Observation percorridas pelo agente."""
    if not reply.steps:
        return
    with st.expander(f"Raciocínio do agente ({len(reply.steps)} etapas)"):
        for number, step in enumerate(reply.steps, start=1):
            st.markdown(f"**Etapa {number}**")
            st.markdown(f"**Thought:** {step.thought or '(sem texto)'}")
            st.markdown(f"**Action:** `{step.tool}` com a entrada `{step.tool_input}`")
            st.text(f"Observation:\n{step.observation}")


try:
    agent = load_agent()
except ConfigurationError as error:
    st.error(str(error))
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

with st.sidebar:
    st.header("Mentor de Estudos")
    st.write(
        "Agente ReAct que monta cronogramas de estudo, calcula os dias disponíveis até a prova, "
        "considera os feriados nacionais e explica conceitos usando a Wikipédia."
    )
    st.caption(f"Modelo: {MODEL_NAME}")
    st.subheader("Exemplos de perguntas")
    for example in EXAMPLE_QUESTIONS:
        st.caption(example)
    if st.button("Nova conversa", use_container_width=True):
        agent.clear_history(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

st.title("Mentor de Estudos")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("reply"):
            render_steps(message["reply"])

if question := st.chat_input("Pergunte sobre o seu plano de estudos"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Raciocinando e consultando as ferramentas"):
            try:
                reply = agent.ask(question, st.session_state.session_id)
            except AgentError as error:
                st.session_state.messages.pop()
                st.error(str(error))
                st.stop()
        st.markdown(reply.answer)
        render_steps(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply.answer, "reply": reply})
