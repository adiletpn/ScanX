"""HTTP-клиент к локальному Ollama.

Работаем напрямую через REST, без официального SDK: нам нужны всего два
эндпоинта, а лишняя зависимость — это лишний риск при установке офлайн.

Отдельное внимание — обработке отказов. Ollama может быть не запущен,
модель может быть не скачана, генерация может затянуться. Ни один из этих
случаев не должен ронять Streamlit: вместо исключения пользователь получает
текст с конкретной командой, которой это чинится.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Iterator

import requests

import config

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Ошибка обращения к Ollama с текстом, готовым к показу пользователю."""


@dataclass(frozen=True)
class OllamaStatus:
    """Состояние локального сервера Ollama."""

    running: bool
    models: tuple[str, ...]
    model_ready: bool
    """Скачана ли модель из config.OLLAMA_MODEL."""

    message: str
    """Человеческое объяснение проблемы. Пустая строка, если всё хорошо."""


def _normalize(name: str) -> str:
    """Приводит имя модели к виду без тега для сравнения.

    Ollama показывает модели как ``gemma3:4b``, а в конфиге может стоять
    и ``gemma3``, и ``gemma3:4b`` — сравниваем по базовому имени.
    """
    return name.split(":")[0].strip().lower()


def check_status(model: str | None = None) -> OllamaStatus:
    """Проверяет, запущен ли Ollama и скачана ли нужная модель.

    Вызывается при открытии вкладки, поэтому таймаут короткий: подвисший
    интерфейс хуже, чем сообщение «сервер не отвечает».
    """
    target = model or config.OLLAMA_MODEL

    try:
        response = requests.get(
            f"{config.OLLAMA_HOST}/api/tags",
            timeout=config.OLLAMA_CONNECT_TIMEOUT_SEC,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Ollama недоступен: %s", exc)
        return OllamaStatus(
            running=False,
            models=(),
            model_ready=False,
            message=(
                "Локальный ИИ не запущен.\n\n"
                "Откройте приложение Ollama или выполните в терминале:\n"
                "`ollama serve`"
            ),
        )

    models = tuple(m.get("name", "") for m in response.json().get("models", []))
    ready = any(_normalize(m) == _normalize(target) for m in models)

    if not ready:
        return OllamaStatus(
            running=True,
            models=models,
            model_ready=False,
            message=(
                f"Модель `{target}` не скачана.\n\n"
                f"Выполните в терминале:\n`ollama pull {target}`"
            ),
        )

    return OllamaStatus(running=True, models=models, model_ready=True, message="")


def encode_image(data: bytes) -> str:
    """Кодирует картинку в base64 — в таком виде Ollama принимает изображения."""
    return base64.b64encode(data).decode("ascii")


def chat_stream(
    messages: list[dict[str, object]],
    model: str | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """Отправляет диалог в Ollama и отдаёт ответ по кускам.

    Стриминг здесь не украшение: локальная модель печатает ответ
    несколько секунд, и без потокового вывода интерфейс выглядит зависшим.

    Args:
        messages: история в формате Ollama — список словарей с ключами
            ``role`` (``system`` / ``user`` / ``assistant``), ``content``
            и опционально ``images`` (список base64-строк).
        model: имя модели; по умолчанию из config.
        temperature: температура генерации.

    Yields:
        Куски текста ответа по мере генерации.

    Raises:
        OllamaError: сервер недоступен, модель не найдена или вышел таймаут.
    """
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": (
                temperature if temperature is not None else config.OLLAMA_TEMPERATURE
            ),
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }

    try:
        with requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json=payload,
            stream=True,
            timeout=(config.OLLAMA_CONNECT_TIMEOUT_SEC, config.OLLAMA_TIMEOUT_SEC),
        ) as response:
            if response.status_code == 404:
                raise OllamaError(
                    f"Модель `{payload['model']}` не найдена. "
                    f"Выполните: `ollama pull {payload['model']}`"
                )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    # Оборванная строка в потоке — пропускаем, но не падаем:
                    # потерять один фрагмент лучше, чем весь ответ.
                    log.debug("Не разобрал строку потока: %r", line[:120])
                    continue

                if chunk.get("error"):
                    raise OllamaError(f"Ollama вернул ошибку: {chunk['error']}")

                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece

                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError as exc:
        log.error("Нет связи с Ollama: %s", exc)
        raise OllamaError(
            "Локальный ИИ не отвечает. Откройте приложение Ollama "
            "или выполните `ollama serve`."
        ) from exc
    except requests.exceptions.ReadTimeout as exc:
        raise OllamaError(
            "Модель отвечает слишком долго. Попробуйте задать вопрос короче."
        ) from exc
    except requests.RequestException as exc:
        log.error("Сбой запроса к Ollama: %s", exc)
        raise OllamaError("Не удалось получить ответ от локальной модели.") from exc


def chat_once(
    messages: list[dict[str, object]],
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Собирает полный ответ целиком.

    Нужен там, где ответ используется как данные, а не показывается
    по мере генерации: триаж и лист для врача разбираются построчно.
    """
    return "".join(chat_stream(messages, model=model, temperature=temperature))


def warm_up(model: str | None = None) -> bool:
    """Прогревает модель — загружает её в память заранее.

    Первый запрос к свежезапущенному Ollama тратит 10–30 секунд на загрузку
    весов. На сцене эта пауза выглядит как зависшее приложение, поэтому
    прогрев делается заранее, пока пользователь читает инструкцию.

    Returns:
        True, если модель ответила и готова.
    """
    try:
        requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": model or config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "ок"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=(config.OLLAMA_CONNECT_TIMEOUT_SEC, 90),
        ).raise_for_status()
        log.info("Модель прогрета")
        return True
    except requests.RequestException as exc:
        log.warning("Прогрев не удался: %s", exc)
        return False
