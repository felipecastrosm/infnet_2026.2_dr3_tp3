import pytest
import requests

import tools
from tools import calcular_dias_de_estudo, consultar_feriados, distribuir_horas, pesquisar_wikipedia

HOLIDAYS_2026 = [
    {"date": "2026-11-20", "name": "Dia da consciência negra", "type": "national"},
    {"date": "2026-12-25", "name": "Natal", "type": "national"},
]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.payload


@pytest.fixture(autouse=True)
def clear_holiday_cache():
    tools.fetch_national_holidays.cache_clear()


@pytest.fixture
def holidays_api(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: FakeResponse(HOLIDAYS_2026))


def test_consultar_feriados_lists_holidays(holidays_api):
    result = consultar_feriados.invoke("2026")
    assert "20/11/2026 (sexta): Dia da consciência negra" in result
    assert "25/12/2026 (sexta): Natal" in result


def test_consultar_feriados_rejects_invalid_year():
    assert "Ano inválido" in consultar_feriados.invoke("ano que vem")


def test_calcular_dias_discounts_holidays_and_days_off(holidays_api):
    result = calcular_dias_de_estudo.invoke("2026-12-21; 2026-12-27; sábado, domingo")
    assert "Dias corridos: 7" in result
    assert "Natal" in result
    assert "Dias disponíveis para estudo: 4" in result


def test_calcular_dias_accepts_brazilian_date_format(holidays_api):
    assert "Dias corridos: 2" in calcular_dias_de_estudo.invoke("01/12/2026; 02/12/2026")


def test_calcular_dias_rejects_inverted_period(holidays_api):
    assert "posterior" in calcular_dias_de_estudo.invoke("2026-12-10; 2026-12-01")


def test_calcular_dias_rejects_unknown_weekday(holidays_api):
    assert "Dia da semana inválido" in calcular_dias_de_estudo.invoke("2026-12-01; 2026-12-10; feriadão")


def test_calcular_dias_reports_network_failure(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(tools.requests, "get", raise_timeout)
    assert "demorou demais" in calcular_dias_de_estudo.invoke("2026-12-01; 2026-12-10")


def test_distribuir_horas_is_proportional():
    result = distribuir_horas.invoke("60; Matemática=3, Português=2, História=1")
    assert "Matemática: 30.0 h (50%)" in result
    assert "História: 10.0 h (17%)" in result


@pytest.mark.parametrize(
    ("raw_input", "expected_error"),
    [
        ("0; Física=1", "maior que zero"),
        ("dez; Física=1", "Total de horas inválido"),
        ("10; Física", "Use o formato nome=peso"),
        ("10; Física=-1", "deve ser maior que zero"),
        ("10", "Use o formato"),
    ],
)
def test_distribuir_horas_validates_input(raw_input, expected_error):
    assert expected_error in distribuir_horas.invoke(raw_input)


def test_pesquisar_wikipedia_returns_summary(monkeypatch):
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Estequiometria",
                    "fullurl": "https://pt.wikipedia.org/wiki/Estequiometria",
                    "extract": "Estequiometria é o cálculo das quantidades de reagentes e produtos.",
                }
            }
        }
    }
    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    result = pesquisar_wikipedia.invoke("estequiometria")
    assert "Artigo: Estequiometria" in result
    assert "reagentes e produtos" in result


def test_pesquisar_wikipedia_without_results(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: FakeResponse({"batchcomplete": ""}))
    assert "Nenhum artigo encontrado" in pesquisar_wikipedia.invoke("xyzabc")


def test_pesquisar_wikipedia_reports_http_error(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: FakeResponse({}, status_code=503))
    assert "erro HTTP 503" in pesquisar_wikipedia.invoke("química")
