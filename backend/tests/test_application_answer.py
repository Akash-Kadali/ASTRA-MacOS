"""Regression tests for the low-cost Jobright application-answer path."""

import asyncio

from backend.api import jobright_agent, talk


def test_agent_domain_allowlist_requires_https_and_known_ats():
    assert jobright_agent._trusted_application_url("https://boards.greenhouse.io/acme/jobs/1")
    assert jobright_agent._trusted_application_url("https://jobs.apple.com/en-us/details/1")
    assert not jobright_agent._trusted_application_url("http://boards.greenhouse.io/acme/jobs/1")
    assert not jobright_agent._trusted_application_url("https://example.invalid/apply")


def test_talk_application_answer_is_grounded_then_superhuman(monkeypatch):
    calls = []

    async def fake_call(system, user, model, temperature=0.4, max_completion_tokens=None):
        calls.append((system, user, model, max_completion_tokens))
        return "I built a Python evaluation workflow and used it to review model errors."

    monkeypatch.setattr(talk, "_call_openai", fake_call)
    request = talk.ApplicationAnswerReq(
        question="Describe relevant Python experience.",
        jd_text="The role needs Python and model evaluation.",
        resume_text="Built a Python evaluation workflow for model errors.",
        profile={"first_name": "Sri Akash"},
        model="test-model",
    )
    result = asyncio.run(talk.application_answer(request))

    assert result["mode"] == "grounded_plus_superhuman"
    assert len(calls) == 2
    assert calls[0][2] == "test-model"
    assert calls[0][3] == 500
    assert "Natural Indian English Rewriting Prompt" in calls[1][1]


def test_jobright_answer_cache_avoids_repeat_call(monkeypatch, tmp_path):
    cache_path = tmp_path / "answers.json"
    monkeypatch.setattr(jobright_agent, "ANSWER_CACHE", cache_path)
    monkeypatch.setattr(jobright_agent, "_pdf_text", lambda _path: "Python resume evidence")
    calls = 0

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answer": "I used Python to build and test ML workflows."}

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(jobright_agent.requests, "post", fake_post)
    job = {"id": "job-1", "description": "Python ML role"}
    profile = {"first_name": "Sri Akash"}
    resume = tmp_path / "resume.pdf"

    first = jobright_agent._hirex_answer("Describe your Python experience", job, resume, profile)
    second = jobright_agent._hirex_answer("Describe your Python experience", job, resume, profile)

    assert first == second
    assert calls == 1


def test_sensitive_questions_never_route_to_hirex(monkeypatch, tmp_path):
    monkeypatch.setattr(
        jobright_agent.requests, "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not call Hirex")),
    )
    answer = jobright_agent._hirex_answer(
        "Will you require visa sponsorship?",
        {"id": "job-1", "description": "Role"},
        tmp_path / "resume.pdf",
        {},
    )
    assert answer == ""


def test_application_field_defaults_cover_greenhouse_identity_fields():
    profile = {
        "first_name": "Sri Akash", "last_name": "Kadali",
        "location": "663 Newell Rd, Palo Alto, CA 94303",
    }
    assert jobright_agent._field_value("Preferred First Name", profile, {}) == "Sri Akash"
    assert jobright_agent._field_value("Preferred Last Name/Surname", profile, {}) == "Kadali"
    assert jobright_agent._field_value("Zip Code/Postal Code", profile, {}) == "94303"


def test_application_select_defaults_follow_audited_profile():
    profile = {
        "authorized_to_work": "Yes", "requires_sponsorship": "Yes",
        "gender": "Male", "veteran": "No",
    }
    assert jobright_agent._select_candidates("Are you eligible to work?", profile) == ["Yes"]
    assert jobright_agent._select_candidates("Will you need immigration sponsorship?", profile) == ["Yes"]
    assert jobright_agent._select_candidates("Are you currently an employee?", profile) == ["No"]
    assert jobright_agent._select_candidates("Gender", profile) == ["Male"]


def test_agent_resume_filename_uses_name_company_and_role():
    filename = jobright_agent._resume_filename({
        "company": "Example/AI", "title": "ML Engineer: Inference",
    })
    assert filename == "Sri Akash Kadali - Example-AI - ML Engineer- Inference.pdf"
