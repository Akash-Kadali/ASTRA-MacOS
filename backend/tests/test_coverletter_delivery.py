import asyncio

from backend.api import coverletter, superhuman


def test_coverletter_always_uses_full_superhuman_prompt(monkeypatch):
    captured = {}

    async def fake_rewrite(text, mode, tone, constraints, latex_safe, max_len):
        captured.update({
            "text": text, "mode": mode, "tone": tone,
            "constraints": constraints, "latex_safe": latex_safe, "max_len": max_len,
        })
        return "I built the service and checked it with the team."

    monkeypatch.setattr(superhuman, "rewrite_single", fake_rewrite)
    result = asyncio.run(coverletter._humanize("Original cover letter body", "balanced"))

    assert result == "I built the service and checked it with the team."
    assert captured["mode"] == "coverletter"
    assert captured["max_len"] == 12000
    assert captured["latex_safe"] is False
    assert captured["constraints"] == {"no_fabrication": True, "keep_metrics": True}


def test_desktop_coverletter_path_deduplicates_and_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    first = coverletter._desktop_coverletter_path("Acme", "ML Engineer", b"%PDF-first")
    same = coverletter._desktop_coverletter_path("Acme", "ML Engineer", b"%PDF-first")
    second = coverletter._desktop_coverletter_path("Acme", "ML Engineer", b"%PDF-second")

    assert first == same
    assert first.name == "Sri Akash Kadali - Acme - ML Engineer - Cover Letter.pdf"
    assert second.name == "Sri Akash Kadali - Acme - ML Engineer - Cover Letter - v2.pdf"
    assert first.parent == tmp_path / "Desktop" / "ASTRA Cover Letters"
