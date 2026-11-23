import sys
import uuid
from datetime import datetime
from pathlib import Path

from agent import AgentError, StudyMentorAgent
from config import MODEL_NAME, ConfigurationError

OUTPUT_FILE = Path(__file__).parent / "evidencias" / "casos_de_uso_resultados.md"

USE_CASES = [
    (
        "UC1: tempo disponível até a prova",
        "Minha prova do ENEM é em 2026-12-20. Vou começar a estudar amanhã, 3 horas por dia, "
        "e não estudo aos domingos. Quantos dias e quantas horas de estudo eu terei?",
    ),
    (
        "UC2: distribuição das horas usando a memória",
        "Divida essas horas entre Matemática, Redação, Biologia e Química. "
        "Matemática é a minha maior dificuldade e Redação vem em seguida.",
    ),
    (
        "UC3: explicação de conceito",
        "Na parte de Química eu não lembro o que é estequiometria. Pode me explicar de forma resumida?",
    ),
    (
        "UC4: feriados e lembrança do contexto",
        "E se a prova fosse adiada para 2027-01-10, mantendo a mesma rotina? "
        "Quais feriados entram no período e quantas horas eu teria?",
    ),
    (
        "UC5: entrada inválida e recuperação",
        "Na verdade quero dividir 0 horas entre Física e Inglês, pode fazer?",
    ),
]


def format_case(title: str, question: str, reply) -> str:
    lines = [f"## {title}", "", f"**Pergunta:** {question}", ""]
    for number, step in enumerate(reply.steps, start=1):
        lines += [
            f"**Etapa {number}**",
            "",
            f"- Thought: {step.thought}",
            f"- Action: `{step.tool}`",
            f"- Action Input: `{step.tool_input}`",
            "",
            "```",
            f"Observation: {step.observation}",
            "```",
            "",
        ]
    lines += ["**Resposta final:**", "", reply.answer, ""]
    return "\n".join(lines)


def main() -> int:
    try:
        agent = StudyMentorAgent()
    except ConfigurationError as error:
        print(error)
        return 1

    session_id = str(uuid.uuid4())
    sections = [
        "# Resultados dos casos de uso",
        "",
        f"Execução em {datetime.now():%d/%m/%Y %H:%M} com o modelo `{MODEL_NAME}`, "
        "todos os casos na mesma sessão de memória.",
        "",
    ]
    for title, question in USE_CASES:
        print(f"Executando {title}")
        try:
            reply = agent.ask(question, session_id)
        except AgentError as error:
            print(f"Falha: {error}")
            return 1
        print(reply.answer, "\n")
        sections.append(format_case(title, question, reply))

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(sections), encoding="utf-8")
    print(f"Resultados salvos em {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
