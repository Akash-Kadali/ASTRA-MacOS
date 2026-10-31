"""Read-only audit of application questions for relevant Jobright roles."""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

from backend.api.jobright import _classify_role

DATA = Path.home() / "Library" / "Application Support" / "ASTRA"
STORE = DATA / "jobright_jobs.json"
PROFILE = DATA / "jobright_browser"
REPORT = DATA / "jobright_question_audit.json"


async def main() -> None:
    payload = json.loads(STORE.read_text(encoding="utf-8"))
    jobs = [j for j in payload["items"] if _classify_role(j.get("title", ""))[1]]
    sem = asyncio.Semaphore(5)
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(str(PROFILE), channel="chrome", headless=True)

        async def resolve(job):
            if job.get("application_url"):
                return
            async with sem:
                page = await context.new_page()
                try:
                    await page.goto(job["url"], wait_until="domcontentloaded", timeout=60000)
                    await page.wait_for_timeout(1200)
                    link = page.locator('a:has-text("Original Job Post")').first
                    if await link.count():
                        job["application_url"] = await link.get_attribute("href") or ""
                except Exception as exc:
                    job["audit_error"] = str(exc)[:300]
                finally:
                    await page.close()

        await asyncio.gather(*(resolve(j) for j in jobs))
        STORE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        async def audit(job):
            url = job.get("application_url")
            if not url:
                return {"job_id": job.get("id"), "title": job.get("title"), "company": job.get("company"),
                        "url": "", "domain": "", "questions": [], "error": "No original application URL"}
            async with sem:
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=75000)
                    await page.wait_for_timeout(1500)
                    rows = await page.locator("input, textarea, select").evaluate_all("""els => els.map(e => {
                      const id=e.id; let label='';
                      if(id) label=(document.querySelector('label[for="'+CSS.escape(id)+'"]')?.innerText||'');
                      if(!label) label=(e.closest('label')?.innerText||'');
                      if(!label) label=e.getAttribute('aria-label')||e.getAttribute('placeholder')||e.name||'';
                      return {label:label.trim().replace(/\\s+/g,' ').slice(0,300), name:e.name||'',
                        type:e.type||e.tagName.toLowerCase(), required:e.required||e.getAttribute('aria-required')==='true'};
                    }).filter(x=>x.label)""")
                    return {"job_id": job.get("id"), "title": job.get("title"), "company": job.get("company"),
                            "url": page.url, "domain": urlparse(page.url).netloc, "questions": rows, "error": ""}
                except Exception as exc:
                    return {"job_id": job.get("id"), "title": job.get("title"), "company": job.get("company"),
                            "url": url, "domain": urlparse(url).netloc, "questions": [], "error": str(exc)[:300]}
                finally:
                    await page.close()

        results = await asyncio.gather(*(audit(j) for j in jobs))
        await context.close()

    normalized = Counter()
    examples = defaultdict(list)
    for result in results:
        for q in result["questions"]:
            label = re.sub(r"\s+", " ", q["label"]).strip(" *:")
            key = re.sub(r"\b(the|your|please)\b", "", label.casefold())
            key = re.sub(r"[^a-z0-9+]+", " ", key).strip()
            if len(key) < 2:
                continue
            normalized[key] += 1
            if len(examples[key]) < 3:
                examples[key].append({"label": label, "company": result["company"], "title": result["title"],
                                      "required": q["required"], "type": q["type"]})
    report = {
        "jobs_audited": len(results),
        "application_urls": sum(bool(x["url"]) for x in results),
        "forms_with_questions": sum(bool(x["questions"]) for x in results),
        "domains": Counter(x["domain"] for x in results if x["domain"]).most_common(),
        "questions": [{"key": k, "count": c, "examples": examples[k]} for k, c in normalized.most_common()],
        "results": results,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("jobs_audited", "application_urls", "forms_with_questions", "domains")}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
