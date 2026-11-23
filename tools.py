import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache

import requests
from langchain_core.tools import ToolException, tool

from config import HTTP_HEADERS, HTTP_TIMEOUT_SECONDS

BRASIL_API_HOLIDAYS_URL = "https://brasilapi.com.br/api/feriados/v1/{year}"
WIKIPEDIA_API_URL = "https://pt.wikipedia.org/w/api.php"
WEEKDAY_NAMES = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
MAX_SUMMARY_LENGTH = 1500


def _normalize(text: str) -> str:
    """Remove acentos, espaços extras e aspas para comparar textos digitados pelo usuário."""
    without_accents = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return without_accents.strip().strip("\"'").lower()


def _split_parts(raw_input: str) -> list[str]:
    """Separa a entrada da ferramenta em partes usando ponto e vírgula."""
    return [part.strip().strip("\"'") for part in raw_input.strip().strip("\"'").split(";")]


def _parse_date(text: str) -> date:
    """Converte datas nos formatos AAAA-MM-DD ou DD/MM/AAAA."""
    for date_format in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text.strip(), date_format).date()
        except ValueError:
            continue
    raise ToolException(f"Data inválida: '{text}'. Use o formato AAAA-MM-DD.")


def _format_date(value: date) -> str:
    return f"{value.strftime('%d/%m/%Y')} ({WEEKDAY_NAMES[value.weekday()]})"


def _fetch_json(url: str, params: dict | None = None) -> dict | list:
    """Faz a requisição HTTP e converte falhas de rede em mensagens que o agente consegue observar."""
    try:
        response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise ToolException("O serviço externo demorou demais para responder. Tente novamente mais tarde.") from exc
    except requests.HTTPError as exc:
        raise ToolException(f"O serviço externo respondeu com erro HTTP {exc.response.status_code}.") from exc
    except requests.RequestException as exc:
        raise ToolException("Não foi possível conectar ao serviço externo.") from exc
    except ValueError as exc:
        raise ToolException("O serviço externo retornou uma resposta inválida.") from exc


@lru_cache(maxsize=16)
def fetch_national_holidays(year: int) -> tuple[tuple[date, str], ...]:
    """Busca na BrasilAPI os feriados nacionais de um ano, mantendo o resultado em cache."""
    if not 1900 <= year <= 2199:
        raise ToolException("A BrasilAPI só possui feriados entre 1900 e 2199.")
    data = _fetch_json(BRASIL_API_HOLIDAYS_URL.format(year=year))
    return tuple((date.fromisoformat(item["date"]), item["name"]) for item in data)


@tool("consultar_feriados")
def consultar_feriados(ano: str) -> str:
    """Consulta os feriados nacionais do Brasil em um ano usando a BrasilAPI.
    Entrada: o ano com quatro dígitos, por exemplo 2026."""
    try:
        year = int(_normalize(ano))
    except ValueError as exc:
        raise ToolException(f"Ano inválido: '{ano}'. Informe apenas o ano, por exemplo 2026.") from exc
    holidays = fetch_national_holidays(year)
    lines = [f"{_format_date(holiday_date)}: {name}" for holiday_date, name in holidays]
    return f"Feriados nacionais de {year}:\n" + "\n".join(lines)


def _parse_days_off(text: str) -> set[int]:
    """Converte nomes de dias da semana, como 'sabado, domingo', em índices de 0 a 6."""
    normalized_names = [_normalize(name) for name in WEEKDAY_NAMES]
    days_off = set()
    for raw_name in text.split(","):
        name = _normalize(raw_name).removesuffix("-feira")
        if not name:
            continue
        if name not in normalized_names:
            raise ToolException(f"Dia da semana inválido: '{raw_name.strip()}'. Use nomes como segunda ou domingo.")
        days_off.add(normalized_names.index(name))
    return days_off


@tool("calcular_dias_de_estudo")
def calcular_dias_de_estudo(periodo: str) -> str:
    """Calcula quantos dias de estudo existem entre duas datas (inclusive), descontando os feriados
    nacionais obtidos na BrasilAPI e os dias da semana em que o estudante não estuda.
    Entrada: data inicial; data final; dias de folga opcionais separados por vírgula.
    Exemplo: 2026-11-24; 2026-12-19; sábado, domingo"""
    parts = _split_parts(periodo)
    if len(parts) < 2:
        raise ToolException("Informe a data inicial e a data final separadas por ponto e vírgula.")
    start_date, end_date = _parse_date(parts[0]), _parse_date(parts[1])
    if end_date < start_date:
        raise ToolException("A data final deve ser igual ou posterior à data inicial.")
    days_off = _parse_days_off(parts[2]) if len(parts) > 2 else set()

    holidays = {}
    for year in range(start_date.year, end_date.year + 1):
        holidays.update(dict(fetch_national_holidays(year)))

    total_days = (end_date - start_date).days + 1
    period_days = [start_date + timedelta(days=offset) for offset in range(total_days)]
    holidays_in_period = [day for day in period_days if day in holidays]
    study_days = [day for day in period_days if day not in holidays and day.weekday() not in days_off]

    holiday_lines = [f"- {_format_date(day)}: {holidays[day]}" for day in holidays_in_period] or ["- nenhum"]
    days_off_text = ", ".join(WEEKDAY_NAMES[index] for index in sorted(days_off)) or "nenhum"
    return (
        f"Período: {_format_date(start_date)} a {_format_date(end_date)}\n"
        f"Dias corridos: {total_days}\n"
        f"Dias de folga semanais: {days_off_text}\n"
        f"Feriados nacionais no período:\n" + "\n".join(holiday_lines) + "\n"
        f"Dias disponíveis para estudo: {len(study_days)}"
    )


@tool("distribuir_horas")
def distribuir_horas(dados: str) -> str:
    """Distribui um total de horas de estudo entre disciplinas de forma proporcional ao peso de cada uma
    (dificuldade ou importância na prova).
    Entrada: total de horas; disciplinas no formato nome=peso separadas por vírgula.
    Exemplo: 60; Matemática=3, Português=2, História=1"""
    parts = _split_parts(dados)
    if len(parts) != 2:
        raise ToolException("Use o formato: total de horas; Disciplina=peso, Disciplina=peso.")
    try:
        total_hours = float(parts[0].replace(",", "."))
    except ValueError as exc:
        raise ToolException(f"Total de horas inválido: '{parts[0]}'.") from exc
    if total_hours <= 0:
        raise ToolException("O total de horas deve ser maior que zero.")

    weights = {}
    for item in parts[1].split(","):
        name, separator, weight_text = item.partition("=")
        if not separator or not name.strip():
            raise ToolException(f"Disciplina inválida: '{item.strip()}'. Use o formato nome=peso.")
        try:
            weight = float(weight_text.strip())
        except ValueError as exc:
            raise ToolException(f"Peso inválido para '{name.strip()}': '{weight_text.strip()}'.") from exc
        if weight <= 0:
            raise ToolException(f"O peso de '{name.strip()}' deve ser maior que zero.")
        weights[name.strip()] = weight

    total_weight = sum(weights.values())
    lines = [
        f"- {name}: {total_hours * weight / total_weight:.1f} h ({weight / total_weight:.0%})"
        for name, weight in sorted(weights.items(), key=lambda entry: entry[1], reverse=True)
    ]
    return f"Distribuição de {total_hours:g} horas:\n" + "\n".join(lines)


@tool("pesquisar_wikipedia")
def pesquisar_wikipedia(termo: str) -> str:
    """Pesquisa um tema na Wikipédia em português e retorna o resumo do artigo mais relevante.
    Útil para explicar conceitos das disciplinas e sugerir tópicos de estudo.
    Entrada: o tema a pesquisar, por exemplo Revolução Francesa."""
    search_term = termo.strip().strip("\"'")
    if not search_term:
        raise ToolException("Informe um tema para pesquisar.")
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrlimit": 1,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
    }
    data = _fetch_json(WIKIPEDIA_API_URL, params)
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return f"Nenhum artigo encontrado na Wikipédia para '{search_term}'."
    page = next(iter(pages.values()))
    summary = page.get("extract", "").strip() or "O artigo não possui resumo."
    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[:MAX_SUMMARY_LENGTH].rsplit(" ", 1)[0] + " [resumo truncado]"
    return f"Artigo: {page.get('title')}\nFonte: {page.get('fullurl')}\nResumo: {summary}"


TOOLS = [consultar_feriados, calcular_dias_de_estudo, distribuir_horas, pesquisar_wikipedia]

for study_tool in TOOLS:
    study_tool.handle_tool_error = True
