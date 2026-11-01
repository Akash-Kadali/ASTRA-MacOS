from backend.api import internet_jobs


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
