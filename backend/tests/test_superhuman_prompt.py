import asyncio

from backend.api import superhuman


def test_superhuman_uses_approved_indian_english_prompt():
    prompt = superhuman.SUPERHUMAN_REWRITE_PROMPT.format(text="Original text")
    assert "IELTS Band 6–6.5" in prompt
    assert "Rewrite from understanding" in prompt
    assert '"leverage", "robust", "seamless", "transformative"' in prompt
    assert "Imperfect is the new perfect" in prompt
    assert "No artificial humanisation" in prompt
    assert prompt.endswith("Original text")


def test_rewrite_single_returns_only_prompted_model_output(monkeypatch):
    captured = []

    async def fake_rewrite(text):
        captured.append(text)
        return "I completed the work and shared the result."

    monkeypatch.setattr(superhuman, "_call_prompted_rewrite", fake_rewrite)
    result = asyncio.run(superhuman.rewrite_single(
        "I have completed the work and have shared the result.",
        "paragraph", "balanced", {}, False, 1600,
    ))

    assert captured == ["I have completed the work and have shared the result."]
    assert result == "I completed the work and shared the result."
