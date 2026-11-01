from backend.api import internet_jobs, jobright_agent


def test_related_us_job_is_ranked_for_bay_area():
    item = internet_jobs._finish({
        "id": "one", "title": "Machine Learning Engineer, LLM Inference",
        "company": "Example AI", "location": "San Francisco, CA",
        "description": "Build and deploy large language model inference systems using Python and PyTorch.",
    })
    assert item is not None
    assert item["region"] == "Bay Area"
    assert item["region_priority"] == 3


def test_phd_only_and_unrelated_jobs_are_excluded():
    assert internet_jobs._finish({
        "id": "phd", "title": "Research Scientist, AI",
        "company": "Example", "location": "California",
        "description": "Required qualification: currently pursuing a PhD in Computer Science.",
    }) is None
    assert internet_jobs._finish({
        "id": "web", "title": "Senior Frontend Engineer",
        "company": "Example", "location": "California",
        "description": "Build React pages for our marketing website.",
    }) is None


def test_newer_jobs_sort_first_within_region():
    older = {"region_priority": 3, "updated_at": "2026-01-01T00:00:00Z", "company": "B"}
    newer = {"region_priority": 3, "updated_at": "2026-10-01T00:00:00Z", "company": "A"}
    assert sorted([older, newer], key=internet_jobs._sort_key)[0] is newer


def test_saved_profile_filters_experience_student_and_clearance(monkeypatch):
    monkeypatch.setattr(jobright_agent, "settings", lambda: {"profile": {
        "years_experience": "2", "currently_student": "no", "security_clearance": "no",
        "requires_sponsorship": "yes", "citizenship": "India",
    }})
    base = {"company": "Example", "location": "San Francisco, CA", "title": "Machine Learning Engineer"}
    assert internet_jobs._finish({**base, "id": "years", "description": "Required: at least three years of ML experience."}) is None
    assert internet_jobs._finish({**base, "id": "student", "description": "Must be currently pursuing a Master's degree."}) is None
    assert internet_jobs._finish({**base, "id": "clearance", "description": "Ability to obtain a Secret security clearance is required."}) is None
    assert internet_jobs._finish({**base, "id": "entry", "description": "Master's degree and one year of Python experience."}) is not None


def test_total_experience_is_not_reduced_by_nested_or_later_requirements(monkeypatch):
    monkeypatch.setattr(jobright_agent, "settings", lambda: {"profile": {
        "years_experience": "2", "currently_student": "no", "security_clearance": "no",
        "requires_sponsorship": "yes", "citizenship": "India",
    }})
    base = {"company": "Example", "location": "San Francisco, CA", "title": "Machine Learning Engineer"}
    description = (
        "Required: 4+ years of total experience, including 2+ years working on large-scale model training platforms. "
        "A completed Master's degree in Computer Science is preferred."
    )
    assert internet_jobs._experience_minimum(description) == 4
    assert internet_jobs._finish({**base, "id": "compound-years", "description": description}) is None


def test_frontier_research_seniority_without_numeric_years_is_excluded(monkeypatch):
    monkeypatch.setattr(jobright_agent, "settings", lambda: {"profile": {
        "years_experience": "2", "currently_student": "no", "security_clearance": "no",
        "requires_sponsorship": "yes", "citizenship": "India",
    }})
    base = {"company": "Example", "location": "San Francisco, CA"}
    assert internet_jobs._finish({
        **base, "id": "opaque", "title": "Research Scientist, Interpretability",
        "description": "Have a strong track record of scientific research and direct interpretability work. "
                       "Minimum years of experience will correlate with the internal job level requirements.",
    }) is None
    assert internet_jobs._finish({
        **base, "id": "new-grad", "title": "Machine Learning Research Scientist, New College Grad",
        "description": "New college graduate role. Master's degree accepted and published ML research preferred.",
    }) is not None
