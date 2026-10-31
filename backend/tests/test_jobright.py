from backend.api.jobright import (
    _already_in_resume_library, _classify_role, _description_sections,
    _company_match, _normalized_name, _normalized_role, _normalized_url, _parse_card,
    _education_ineligible, _posted_age_minutes, _requires_current_student,
    _requires_phd, _role_match, _text_tokens,
)


def test_parses_personalized_jobright_card():
    text = """company-logo
just now
Early applicant
Software Engineer - New Grad
Chalk
/
Artificial Intelligence (AI) · Big Data · Growth Stage
San Francisco, CA
Full-time
$170K/yr - $180K/yr
Onsite
New Grad
Why this job is a match
Chalk is hiring new-grad Software Engineers to build its production AI platform.
90%
Experience Level
80%
GOOD MATCH
H1B Sponsor Likely"""
    job = _parse_card("https://jobright.ai/jobs/info/abc123?from=homepage", text)
    assert job is not None
    assert job["id"] == "abc123"
    assert job["title"] == "Software Engineer - New Grad"
    assert job["company"] == "Chalk"
    assert job["posted"] == "just now"
    assert job["location"] == "San Francisco, CA"
    assert job["summary"].startswith("Chalk is hiring")


def test_rejects_marketing_cta():
    assert _parse_card("https://jobright.ai/jobs/info/x", "SIGN IN\nJOIN NOW") is None


def test_resume_library_match_prefers_exact_url_then_company_role():
    job = {
        "url": "https://jobright.ai/jobs/info/abc?from=homepage",
        "application_url": "https://jobs.example.com/roles/42?source=jobright",
        "company": "Example, Inc.",
        "title": "Machine-Learning Engineer",
    }
    assert _normalized_url(job["url"]) == "https://jobright.ai/jobs/info/abc"
    assert _normalized_name(job["company"]) == "example inc"
    assert _already_in_resume_library(job, {"https://jobs.example.com/roles/42"}, set())
    assert _already_in_resume_library(job, set(), {("example inc", "machine learning engineer")})
    assert not _already_in_resume_library(job, set(), {("example inc", "software engineer")})


def test_posted_age_supports_jobright_posted_and_reposted_labels():
    assert _posted_age_minutes("Just now") == 0
    assert _posted_age_minutes("Reposted 4 hours ago") == 240
    assert _posted_age_minutes("1 day ago") == 1440
    assert _posted_age_minutes("Reposted 2 weeks ago") == 20160
    assert _posted_age_minutes("") > _posted_age_minutes("3 months ago")


def test_role_matching_ignores_remote_workplace_qualifier_only():
    assert _normalized_role("[Remote] Machine Learning Engineer") == "machine learning engineer"
    assert _normalized_role("Machine Learning Engineer - Remote") == "machine learning engineer"
    assert _normalized_role("Senior Machine Learning Engineer") != "machine learning engineer"


def test_company_and_role_substring_matching_has_length_guards():
    assert _company_match("oracle", "oracle corporation")
    assert _role_match("machine learning engineer", "remote machine learning engineer")
    assert _role_match("software engineer ai ml", "ai ml software engineer")
    assert not _company_match("ai", "ai systems")
    assert not _role_match("engineer", "machine learning engineer")


def test_saved_jd_can_identify_same_company_job_when_title_varies():
    shared = " ".join(f"skill{i}" for i in range(40))
    job = {"company": "Oracle Corporation", "title": "ML Engineer II", "description": shared + " newdetail"}
    saved = [("oracle", _text_tokens(shared + " savedetail"))]
    assert _already_in_resume_library(job, set(), set(), saved)


def test_phd_specific_required_qualification_is_filtered():
    job = {"title": "Research Scientist", "description_sections": {"required":
        "Pursuing PhD in Computer Science or a related field (or equivalent experience)"}}
    assert _requires_phd(job)
    assert _requires_phd({"title": "Software Engineer, PhD, Early Career"})


def test_phd_preferred_or_lower_degree_alternative_is_allowed():
    assert not _requires_phd({"title": "Software Engineer - New Grad / PhD Welcome"})
    assert not _requires_phd({"title": "ML Engineer", "description_sections": {
        "required": "Bachelor's, Master's, or PhD in Computer Science",
        "preferred": "PhD with publications preferred",
    }})


def test_current_student_degree_requirements_are_filtered():
    examples = (
        "Working toward a bachelor's degree in Computer Science",
        "Currently pursuing a Master's degree in Engineering",
        "Currently enrolled in a PhD program",
        "Must be enrolled at an accredited university",
        "Expected graduation between December 2026 and June 2027",
        "Must be returning to school after the internship",
    )
    for required in examples:
        assert _requires_current_student({"title": "Engineering Intern", "description_sections": {"required": required}})
    assert _education_ineligible({"title": "Machine Learning Co-Op", "description_sections": {}})


def test_completed_degree_and_new_grad_roles_remain_eligible():
    job = {"title": "Machine Learning Engineer - New Grad", "description_sections": {
        "required": "Bachelor's or Master's degree in Computer Science, completed by start date"}}
    assert not _requires_current_student(job)
    assert not _education_ineligible(job)
    assert not _requires_phd({"title": "ML Engineer", "description_sections": {
        "required": "Master's degree in Computer Science",
        "preferred": "PhD preferred",
    }})


def test_role_relevance_categories():
    assert _classify_role("Machine Learning Engineer") == ("Machine Learning & AI", True)
    assert _classify_role("LLM Inference Engineer") == ("LLM & Inference", True)
    assert _classify_role("Computer Vision Engineer") == ("Computer Vision", True)
    assert _classify_role("Software Engineer II") == ("Software Engineering", True)
    assert _classify_role("Senior Full Stack Engineer") == ("Other", False)
    assert _classify_role("Cybersecurity Engineer") == ("Other", False)
    assert _classify_role("Research Engineer, Agentic Security for Gemini") == ("Other", False)


def test_extracts_clean_description_sections():
    text = """Overview
Responsibilities
Build reliable ML services
Ship model evaluation
Qualification
Required
Python
Distributed systems
Preferred
LLM evaluation
Benefits
Visa sponsorship
Company
Acme"""
    sections = _description_sections(text)
    assert sections["responsibilities"] == "Build reliable ML services\nShip model evaluation"
    assert sections["required"] == "Python\nDistributed systems"
    assert sections["preferred"] == "LLM evaluation"
    assert sections["benefits"] == "Visa sponsorship"
