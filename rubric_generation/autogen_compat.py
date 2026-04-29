"""Helpers for AutoGen AgentChat + OpenAIChatCompletionClient."""
from __future__ import annotations

import asyncio
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def make_openai_chat_client(
    *,
    model: str,
    api_key: str | None,
    temperature: float | None = None,
    seed: int | None = None,
) -> OpenAIChatCompletionClient:
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    return OpenAIChatCompletionClient(**kwargs)


async def run_assistant_once(
    *,
    name: str,
    system_message: str | None,
    model: str,
    api_key: str | None,
    task: str,
    temperature: float | None = None,
    seed: int | None = None,
) -> str:
    model_client = make_openai_chat_client(
        model=model, api_key=api_key, temperature=temperature, seed=seed
    )
    try:
        agent = AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=system_message,
        )
        result = await agent.run(task=task)
        if not result.messages:
            return ""
        # Per AutoGen docs/examples, the last message contains the assistant output.
        return str(result.messages[-1].content)
    finally:
        await model_client.close()


def run_assistant_sync(**kwargs: Any) -> str:
    return asyncio.run(run_assistant_once(**kwargs))
