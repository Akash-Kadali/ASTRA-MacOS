from backend.api import internet_jobs, jobright_agent


def test_google_embedded_payload_parser():
    page = """<script>AF_initDataCallback({key: 'ds:1', hash: 'x', data:[[[\"123\",\"Machine Learning Engineer\"]],null,1,1], sideChannel: {}});</script>"""
    assert internet_jobs._google_payload_jobs(page) == [["123", "Machine Learning Engineer"]]
    assert internet_jobs._google_payload_jobs("no payload") == []


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


def test_filtered_jobs_can_be_retained_with_exact_evidence(monkeypatch):
    monkeypatch.setattr(jobright_agent, "settings", lambda: {"profile": {
        "years_experience": "2", "currently_student": "no", "security_clearance": "no",
        "requires_sponsorship": "yes", "citizenship": "India",
    }})
    item = internet_jobs._finish({
        "id": "retained", "company": "Example", "location": "San Francisco, CA",
        "title": "Machine Learning Engineer", "description": "Required: 5+ years of production ML experience.",
    }, retain_filtered=True)
    assert item is not None
    assert item["filter_decision"]["eligible"] is False
    assert item["filter_decision"]["reasons"][0]["code"] == "experience"
    assert "5+ years" in item["filter_decision"]["reasons"][0]["evidence"]


def test_canonical_key_merges_requisition_across_sources():
    first = {"company": "Example AI, Inc.", "title": "ML Engineer", "requisition_id": "REQ-123"}
    second = {"company": "Example AI", "title": "Machine Learning Engineer", "requisition_id": "REQ 123"}
    assert internet_jobs._canonical_key(first) == internet_jobs._canonical_key(second)


def test_year_requirement_with_in_phrase_is_parsed():
    assert internet_jobs._experience_minimum(
        "Requirements:\n3+ years in AI performance engineering, with significant time leading projects."
    ) == 3


def test_qualitative_seniority_without_early_career_path_is_excluded(monkeypatch):
    monkeypatch.setattr(jobright_agent, "settings", lambda: {"profile": {
        "years_experience": "2", "currently_student": "no", "security_clearance": "no",
        "requires_sponsorship": "yes", "citizenship": "India",
    }})
    assert internet_jobs._finish({
        "id": "qualitative", "company": "Example", "location": "San Jose, CA",
        "title": "AI Engineer, Perception",
        "description": "Requirements: Extensive experience in production-level ML engineering.",
    }) is None


def test_sponsor_registry_has_300_targets_and_live_feeds():
    registry = internet_jobs._sponsor_registry()
    assert len(registry["companies"]) == 300
    assert sum(bool(x["direct_feed"]) for x in registry["companies"]) >= 150
    assert len(internet_jobs._configured_feeds()) >= 150
    directory = internet_jobs.companies(q="")
    by_name = {x["company"]: x for x in directory["items"]}
    assert by_name["Google LLC"]["career_url"].startswith("https://www.google.com/about/careers")
    assert by_name["Microsoft Corporation"]["career_url"].startswith("https://jobs.careers.microsoft.com")
    assert by_name["Apple Inc."]["career_url"].startswith("https://jobs.apple.com")
    sources = {(source, company) for source, company, _, _ in internet_jobs._configured_feeds()}
    assert ("Amazon Jobs", "Amazon") in sources
    assert ("Workday", "NVIDIA") in sources


def test_wide_feed_non_engineering_ai_titles_are_excluded(monkeypatch):
    monkeypatch.setattr(jobright_agent, "settings", lambda: {"profile": {
        "years_experience": "2", "currently_student": "no", "security_clearance": "no",
        "requires_sponsorship": "yes", "citizenship": "India",
    }})
    base = {"company": "Example", "location": "San Francisco, CA", "description": "Work with AI products."}
    for title in ("Developer Advocate - AI", "Vice President, Product Management, Managed AI",
                  "Applied AI Engineer, Fullstack", "Applied Scientist Intern"):
        assert internet_jobs._finish({**base, "id": title, "title": title}) is None
    assert internet_jobs._finish({**base, "id": "rs", "title": "AI Research Scientist"}) is None


def test_role_specific_sponsorship_confidence_overrides_company_history():
    sponsor = {"tech_lca_positions": 100}
    unavailable = internet_jobs._sponsorship_assessment(
        "We are unable to provide visa sponsorship for this position.", sponsor)
    assert unavailable["status"] == "unavailable"
    assert unavailable["confidence"] == 0
    explicit = internet_jobs._sponsorship_assessment(
        "Immigration sponsorship support is available for qualified candidates.", {})
    assert explicit["status"] == "posting_supports"
    assert internet_jobs._sponsorship_assessment("Build ML systems.", sponsor)["status"] == "strong_history"


def test_application_value_has_explainable_components():
    item = {"title": "Machine Learning Engineer I", "region_priority": 3,
            "posted_at": internet_jobs._now(), "filter_decision": {
                "eligible": True, "experience_required": 1, "technical_fit": 80,
            }, "sponsorship": {"confidence": 80}}
    score = internet_jobs._rank(item)
    assert score >= 80
    assert {"experience", "technical", "sponsorship", "freshness", "location"} <= set(item["rank_components"])


def test_saved_search_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(internet_jobs, "SETTINGS_STORE", tmp_path / "internet-settings.json")
    saved = internet_jobs.save_search({"name": "Bay AI", "q": "inference", "region": "Bay Area",
                                       "category": "LLM & Inference", "minimum_score": 75})
    assert internet_jobs.saved_searches()["items"][0]["name"] == "Bay AI"
    assert internet_jobs.delete_search(saved["id"])["deleted"] == 1
