"""Behavior-preserving runtime regression tests for the ASTRA optimizer."""

import asyncio
from collections import OrderedDict

from backend.api import optimize


def test_request_state_is_isolated_between_async_tasks():
    async def worker(verb: str, position: int):
        token = optimize._request_state.set(optimize.OptimizationRequestState())
        try:
            state = optimize._state()
            state.used_verbs.add(verb)
            state.keyword_assignments[verb] = position
            await asyncio.sleep(0)
            return set(state.used_verbs), dict(state.keyword_assignments)
        finally:
            optimize._request_state.reset(token)

    async def run():
        return await asyncio.gather(worker("built", 1), worker("tested", 2))

    first, second = asyncio.run(run())
    assert first == ({"built"}, {"built": 1})
    assert second == ({"tested"}, {"tested": 2})


def test_shared_lru_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(optimize, "_SHARED_CACHE_MAX", 3)
    cache = OrderedDict()
    for index in range(5):
        optimize._cache_put(cache, str(index), index)

    assert list(cache.items()) == [("2", 2), ("3", 3), ("4", 4)]
    assert optimize._cache_get(cache, "2") == 2
    assert list(cache) == ["3", "4", "2"]


def test_stable_cache_key_normalizes_case_and_whitespace():
    first = optimize._stable_cache_key(" NVIDIA ", "Software  Engineer", "Python\nCUDA")
    second = optimize._stable_cache_key("nvidia", "software engineer", "python cuda")
    assert first == second


def test_gpt_json_uses_async_client(monkeypatch):
    class Message:
        content = '{"ok": true}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        async def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    async def fake_client():
        return Client()

    monkeypatch.setattr(optimize, "get_openai_client", fake_client)
    monkeypatch.setattr(optimize, "_openai_semaphore", asyncio.Semaphore(1))
    result = asyncio.run(optimize.gpt_json("prompt", model="test-model"))
    assert result == {"ok": True}


def test_gpt_json_singleflight_deduplicates_identical_calls(monkeypatch):
    calls = 0

    async def fake_perform(prompt, temperature=0.0, model="test-model"):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"prompt": prompt}

    monkeypatch.setattr(optimize, "_perform_gpt_json", fake_perform)

    async def run():
        token = optimize._request_state.set(optimize.OptimizationRequestState())
        try:
            results = await asyncio.gather(
                optimize.gpt_json("same prompt", model="test-model"),
                optimize.gpt_json("same prompt", model="test-model"),
            )
            return results, optimize._state().singleflight_hits
        finally:
            optimize._request_state.reset(token)

    results, hits = asyncio.run(run())
    assert results == [{"prompt": "same prompt"}, {"prompt": "same prompt"}]
    assert calls == 1
    assert hits == 1


def test_gpt_json_caches_completed_deterministic_call(monkeypatch):
    calls = 0

    async def fake_perform(prompt, temperature=0.0, model="test-model"):
        nonlocal calls
        calls += 1
        return {"prompt": prompt}

    monkeypatch.setattr(optimize, "_perform_gpt_json", fake_perform)

    async def run():
        token = optimize._request_state.set(optimize.OptimizationRequestState())
        try:
            first = await optimize.gpt_json("same deterministic prompt", model="test-model")
            second = await optimize.gpt_json("same deterministic prompt", model="test-model")
            return first, second, optimize._state().completed_cache_hits
        finally:
            optimize._request_state.reset(token)

    first, second, hits = asyncio.run(run())
    assert first == second
    assert calls == 1
    assert hits == 1


def test_skill_validation_batches_candidates(monkeypatch):
    calls = 0

    async def fake_gpt(prompt, temperature=0.0, model="gpt-5.4-nano"):
        nonlocal calls
        calls += 1
        return {"decisions": [
            {"skill": "Python", "is_skill": True},
            {"skill": "PyTorch", "is_skill": True},
            {"skill": "generic", "is_skill": False},
        ]}

    monkeypatch.setattr(optimize, "gpt_json", fake_gpt)

    async def run():
        token = optimize._request_state.set(optimize.OptimizationRequestState())
        try:
            return await optimize.filter_valid_skills(
                ["Python", "PyTorch", "generic"], "Python and PyTorch are required")
        finally:
            optimize._request_state.reset(token)

    assert asyncio.run(run()) == ["Python", "PyTorch"]
    assert calls == 1
