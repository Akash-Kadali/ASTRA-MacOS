/* Shared application shell for every ASTRA workspace. */
(() => {
  const pages = [
    ["Workspace", "/index.html", "⌁", "Resume Studio"],
    [null, "/dashboard.html", "▦", "Applications"],
    [null, "/genie.html", "✦", "Genie"],
    [null, "/talk.html", "◎", "Talk to HIREX"],
    ["Documents", "/preview.html", "▤", "Preview"],
    [null, "/coverletter.html", "◇", "Cover Letters"],
    ["Intelligence", "/mastermind.html", "◉", "MasterMind"],
    [null, "/superhuman.html", "◈", "SuperHuman"],
  ];
  const titles = Object.fromEntries(pages.filter(x => x[1]).map(x => [x[1], x[3]]));

  function normalizedPath() {
    const path = location.pathname === "/" ? "/index.html" : location.pathname;
    return titles[path] ? path : "/index.html";
  }

  function navMarkup(list, current) {
    return pages.map(([group, href, icon, label]) => {
      const heading = group ? (list ? `<li class="nav-group-label">${group}</li>` : `<div class="nav-label">${group}</div>`) : "";
      const active = href === current;
      const link = `<a href="${href}" class="${active ? (list ? "active-link" : "active") : ""}"${active ? ' aria-current="page"' : ""}><span class="nav-ico">${icon}</span><span>${label}</span></a>`;
      return heading + (list ? `<li>${link}</li>` : link);
    }).join("");
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (new URLSearchParams(location.search).get("embed") === "1") {
      document.body.classList.add("astra-embedded");
      if (new URLSearchParams(location.search).get("theme") === "dark") document.body.classList.add("astra-embed-dark");
      return;
    }
    const current = normalizedPath();
    const oldNav = document.querySelector(".vnav");
    const newNav = document.querySelector(".side .nav");
    if (oldNav) oldNav.innerHTML = navMarkup(true, current);
    if (newNav) newNav.innerHTML = navMarkup(false, current);

    document.querySelectorAll(".brand-name").forEach(el => { el.textContent = "ASTRA"; });

    if (document.querySelector(".side") && !document.querySelector(".astra-unified-topbar")) {
      document.body.classList.add("astra-modern-shell");
      const bar = document.createElement("div");
      bar.className = "astra-unified-topbar";
      bar.setAttribute("role", "status");
      bar.innerHTML = `<div class="topbar-left"><span class="sys-online">Online</span><span class="topbar-tag">Module <b>${titles[current]}</b></span></div><div class="topbar-right"><span class="topbar-tag">ASTRA <b>Career OS</b></span></div>`;
      document.body.appendChild(bar);
    }

    /* Re-append the design system so it always follows page-local legacy CSS. */
    const source = "/static/css/astra-design-system.css?v=3.5.0";
    if (!document.querySelector(`link[href="${source}"]`)) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = source;
      document.head.appendChild(link);
    }
  });
})();
