import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "motion/react";
import {
  Activity,
  BriefcaseBusiness,
  CalendarClock,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleUserRound,
  Command,
  Copy,
  Database,
  Download,
  Edit3,
  ExternalLink,
  FileCheck2,
  FileText,
  FolderOpen,
  Globe2,
  LayoutDashboard,
  Maximize2,
  Menu,
  MessageCircleMore,
  Moon,
  MoreHorizontal,
  RefreshCw,
  Search,
  Sparkles,
  Sun,
  Upload,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";

type Route =
  | "studio"
  | "jobright"
  | "internet"
  | "decisions"
  | "operations"
  | "applications"
  | "genie"
  | "talk"
  | "preview"
  | "cover"
  | "mastermind"
  | "superhuman";
type ApplicationFile = {
  path: string;
  created_at?: string;
  modified_at?: string;
  size?: number;
};
type TimelineEvent = {
  id: string;
  type: string;
  at: string;
  note?: string;
  person?: string;
  attachment?: string;
};
type Application = {
  id: string;
  company: string;
  role: string;
  status: string;
  applied_at?: string;
  call_at?: string;
  interview_at?: string;
  offer_at?: string;
  rejected_at?: string;
  follow_up_at?: string;
  source?: string;
  job_url?: string;
  notes?: string;
  files?: ApplicationFile[];
  timeline?: TimelineEvent[];
};
type ApplicationsResponse = {
  items: Application[];
  total: number;
  page: number;
  limit: number;
  stats: {
    total: number;
    companies: number;
    calls: number;
    interviews: number;
    offers: number;
    response_rate: number;
    status_counts?: Record<string, number>;
  };
};
type Context = {
  key: string;
  company: string;
  role: string;
  updated_at?: string;
  has_humanized?: boolean;
  has_pdf?: boolean;
  has_optimized_pdf?: boolean;
  has_humanized_pdf?: boolean;
  title?: string;
  resume_tex?: string;
  pdf_base64?: string;
  humanized_tex?: string;
  pdf_base64_humanized?: string;
  fit_score?: number | string;
  coverage_ratio?: number | string;
};
type OptimizeResult = {
  company?: string;
  company_name?: string;
  role?: string;
  tex_string: string;
  pdf_base64: string;
  pdf_base64_humanized?: string;
  humanized_tex?: string;
  coverage_ratio?: number;
  rating_score?: number;
  coverage_present?: string[];
  coverage_missing?: string[];
  trim_summary?: { items_removed?: number; final_pages?: number };
};
type CoverResult = {
  company: string;
  role: string;
  tex_string: string;
  body_text?: string;
  pdf_base64: string;
  word_count: number;
  context_key: string;
  salutation_used?: string;
  company_type?: string;
};
type JobrightJob = {
  id: string;
  title: string;
  company: string;
  location?: string;
  posted?: string;
  match?: string;
  url: string;
  application_url?: string;
  card_text?: string;
  summary?: string;
  description?: string;
  description_sections?: Record<string, string>;
  description_status?: string;
  captured_at?: string;
  last_opened_at?: string;
  category?: string;
  relevant?: boolean;
  phd_specific?: boolean;
  education_ineligible?: boolean;
};
type JobrightResponse = {
  account: string;
  updated_at?: string;
  items: JobrightJob[];
  total: number;
  all_total: number;
  relevant_total: number;
  category_counts: Record<string, number>;
  state: { running: boolean; phase: string; message?: string; error?: string };
};
type InternetJob = JobrightJob & {
  source: string;
  region: "Bay Area" | "California" | "United States";
  region_priority: number;
  posted_at?: string;
  updated_at?: string;
  requisition_id?: string;
  salary?: unknown;
  workplace_type?: string;
  rank_score?: number;
  duplicate_count?: number;
  source_names?: string[];
  link_status?: "open" | "closed" | "stale" | "unknown" | "missing" | "not_checked";
  sponsor_history?: boolean;
  tech_lca_positions?: number;
  sponsorship_evidence?: string;
  sponsorship?: { status: string; confidence: number; reason: string };
  rank_components?: Record<string, number>;
  application_policy?: { cooldown_days: number; duplicate_family: boolean };
  data_quality?: { valid_url: boolean; complete_description: boolean; valid_date: boolean };
  filter_decision?: { eligible: boolean; score: number; overridden?: boolean; experience_required?: number; technical_fit?: number; matched_skills?: string[]; missing_skills?: string[]; reasons: { code: string; message: string; evidence: string }[] };
};
type InternetJobsResponse = {
  items: InternetJob[];
  total: number;
  eligible_total: number;
  filtered_total: number;
  new_jobs: number;
  reason_counts: Record<string, number>;
  alerts?: { id: string; name: string; count: number; created_at: string }[];
  sponsor_registry: { targets: number; direct_feeds: number; connected_feeds: number; evidence?: string; source_url?: string; disclaimer?: string };
  counts: Record<string, number>;
  updated_at?: string;
  sources: { source: string; company: string; ok: boolean; jobs: number; error?: string; cached?: boolean; duration_ms?: number; last_success?: string; consecutive_failures?: number }[];
  runtime: { running: boolean; message: string; error?: string; last_refresh?: string };
  active_filters: { years_experience: string; currently_student: string; security_clearance: string; requires_sponsorship: string; degree: string };
};
type AgentProfile = {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  location: string;
  linkedin: string;
  portfolio: string;
  authorized_to_work: string;
  requires_sponsorship: string;
  years_experience: string;
};
type AgentResponse = {
  settings: {
    enabled: boolean;
    run_at: string;
    daily_limit: number;
    auto_submit: boolean;
    profile: AgentProfile;
  };
  runtime: {
    running: boolean;
    current_job?: string;
    message: string;
    last_run?: string;
  };
  events: {
    at: string;
    company: string;
    title: string;
    status: string;
    message: string;
  }[];
};
type DecisionItem = JobrightJob & {
  decision: "actionable" | "review" | "applied" | "ineligible" | "hidden";
  decision_reason: string;
  duplicate_count?: number;
  duplicate_ids?: string[];
  match_details?: {
    application_id?: string;
    company?: string;
    role?: string;
    status?: string;
    pdf_path?: string;
    company_similarity: number;
    role_similarity: number;
    jd_similarity: number;
    confidence: number;
    reason: string;
  };
  eligibility: {
    status: string;
    score: number;
    reasons: string[];
    evidence: string[];
  };
  timeline?: { at: string; action: string; note?: string }[];
};
type DecisionResponse = {
  items: DecisionItem[];
  total: number;
  raw_total?: number;
  counts: Record<string, number>;
  saved_views: { id: string; name: string; filters: Record<string, string> }[];
};

const routes: {
  id: Route;
  label: string;
  description: string;
  icon: typeof Command;
  group: "Workspace" | "Documents" | "Intelligence";
}[] = [
  {
    id: "studio",
    label: "Resume Studio",
    description: "Tailor a résumé",
    icon: Command,
    group: "Workspace",
  },
  {
    id: "jobright",
    label: "Jobright Jobs",
    description: "Import your matched jobs",
    icon: BriefcaseBusiness,
    group: "Workspace",
  },
  {
    id: "internet",
    label: "Internet Jobs",
    description: "Search public US employer feeds",
    icon: Globe2,
    group: "Workspace",
  },
  {
    id: "decisions",
    label: "Decision Centre",
    description: "Review every job decision",
    icon: Check,
    group: "Workspace",
  },
  {
    id: "operations",
    label: "Agent Operations",
    description: "Review, recover and repair",
    icon: Activity,
    group: "Workspace",
  },
  {
    id: "applications",
    label: "Applications",
    description: "Track every opportunity",
    icon: LayoutDashboard,
    group: "Workspace",
  },
  {
    id: "genie",
    label: "Genie",
    description: "Daily command centre",
    icon: Sparkles,
    group: "Workspace",
  },
  {
    id: "talk",
    label: "Talk to HIREX",
    description: "Application answers",
    icon: MessageCircleMore,
    group: "Workspace",
  },
  {
    id: "preview",
    label: "Preview",
    description: "Review generated files",
    icon: FileCheck2,
    group: "Documents",
  },
  {
    id: "cover",
    label: "Cover Letters",
    description: "Create tailored letters",
    icon: FileText,
    group: "Documents",
  },
  {
    id: "mastermind",
    label: "MasterMind",
    description: "Career intelligence",
    icon: WandSparkles,
    group: "Intelligence",
  },
  {
    id: "superhuman",
    label: "SuperHuman",
    description: "Humanize your writing",
    icon: Zap,
    group: "Intelligence",
  },
];
const routeFromHash = (): Route => {
  const value = location.hash.replace("#/", "") as Route;
  return routes.some((r) => r.id === value) ? value : "genie";
};
const legacy: Record<string, string> = {
  studio: "/index.html?embed=1",
  preview: "/preview.html?embed=1",
  cover: "/coverletter.html?embed=1",
};
const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const res = await fetch(path, init);
  if (!res.ok) throw new Error((await res.text()) || "Request failed");
  return res.json();
};
const fetchApplicationArchive = async () => {
  const first = await api<ApplicationsResponse>(
    "/api/applications?limit=250&page=1",
  );
  const pages = Math.ceil(first.total / 250);
  if (pages <= 1) return first.items;
  const rest = await Promise.all(
    Array.from({ length: pages - 1 }, (_, i) =>
      api<ApplicationsResponse>(`/api/applications?limit=250&page=${i + 2}`),
    ),
  );
  return [first, ...rest].flatMap((x) => x.items);
};
const fmt = (v?: string) => {
  if (!v) return "Not recorded";
  const date = new Date(v);
  return Number.isNaN(date.getTime()) ? v : new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      }).format(date);
};
const localDate = (v?: string) =>
  v ? new Date(v).toISOString().slice(0, 16) : "";
const notify = (
  message: string,
  tone: "success" | "error" | "info" = "success",
) =>
  window.dispatchEvent(
    new CustomEvent("astra:toast", { detail: { message, tone } }),
  );

function Sidebar({
  route,
  open,
  onClose,
}: {
  route: Route;
  open: boolean;
  onClose: () => void;
}) {
  const groups = ["Workspace", "Documents", "Intelligence"] as const;
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
      <div className="brand">
        <img src="/static/assets/logo.png" alt="ASTRA" />
        <div>
          <strong>ASTRA</strong>
          <span>Career operating system</span>
        </div>
        <button
          className="mobile-close"
          onClick={onClose}
          aria-label="Close navigation"
        >
          <X size={18} />
        </button>
      </div>
      <nav>
        {groups.map((group) => (
          <div className="nav-group" key={group}>
            <div className="nav-label">{group}</div>
            {routes
              .filter((x) => x.group === group)
              .map((item) => {
                const Icon = item.icon,
                  active = item.id === route;
                return (
                  <a
                    key={item.id}
                    className={active ? "active" : ""}
                    href={`#/${item.id}`}
                    onClick={onClose}
                    aria-current={active ? "page" : undefined}
                  >
                    <Icon size={18} />
                    <span>{item.label}</span>
                    {active && (
                      <motion.span
                        layoutId="active-dot"
                        className="active-dot"
                      />
                    )}
                  </a>
                );
              })}
          </div>
        ))}
      </nav>
      <div className="profile">
        <div className="avatar">SK</div>
        <div>
          <strong>Sri Akash</strong>
          <span>Application workspace</span>
        </div>
        <CircleUserRound size={18} />
      </div>
    </aside>
  );
}

function Header({
  route,
  onMenu,
  dark,
  onTheme,
  onSearch,
}: {
  route: Route;
  onMenu: () => void;
  dark: boolean;
  onTheme: () => void;
  onSearch: () => void;
}) {
  const meta = routes.find((r) => r.id === route)!;
  return (
    <header className="topbar">
      <button className="menu" onClick={onMenu} aria-label="Open navigation">
        <Menu size={20} />
      </button>
      <div className="crumb">
        <span>Workspace</span>
        <ChevronRight size={14} />
        <b>{meta.label}</b>
      </div>
      <div className="topbar-actions">
        <button
          className="global-search-button"
          onClick={onSearch}
          aria-label="Search workspace"
        >
          <Search size={15} />
          <span>Search workspace</span>
          <kbd>⌘ K</kbd>
        </button>
        <div className="system">
          <i /> System online
        </div>
        <button
          className="theme-toggle"
          onClick={onTheme}
          aria-label={dark ? "Use light appearance" : "Use dark appearance"}
          title={dark ? "Use light appearance" : "Use dark appearance"}
        >
          {dark ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </header>
  );
}

function PageTitle({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="page-title">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action && <div className="page-actions">{action}</div>}
    </div>
  );
}

function Skeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="skeletons">
      {Array.from({ length: rows }, (_, i) => (
        <div className="skeleton" key={i} />
      ))}
    </div>
  );
}
function Empty({
  title,
  copy,
  action,
}: {
  title: string;
  copy: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Search size={20} />
      </div>
      <h3>{title}</h3>
      <p>{copy}</p>
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
}

function DecisionCentre() {
  const [view, setView] = useState("actionable");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<DecisionItem | null>(null);
  const [compare, setCompare] = useState<string[]>([]);
  const [limit, setLimit] = useState(80);
  const [confirming, setConfirming] = useState<DecisionItem | null>(null);
  const [requisition, setRequisition] = useState("");
  const [lastAction, setLastAction] = useState<{
    job: DecisionItem;
    action: string;
  } | null>(null);
  const qc = useQueryClient();
  useEffect(() => setLimit(80), [view, query]);
  const decisions = useQuery({
    queryKey: ["decision-centre", view, query, limit],
    queryFn: () =>
      api<DecisionResponse>(
        `/api/decision-center?view=${view}&q=${encodeURIComponent(query)}&limit=${limit}`,
      ),
  });
  const analytics = useQuery({
    queryKey: ["decision-analytics"],
    queryFn: () =>
      api<{
        stats: {
          total: number;
          calls: number;
          interviews: number;
          offers: number;
          response_rate: number;
        };
      }>("/api/decision-center/analytics"),
  });
  const integrity = useQuery({
    queryKey: ["decision-integrity", selected?.id],
    queryFn: () =>
      api<{
        has_match: boolean;
        available: boolean;
        valid_pdf: boolean;
        size: number;
        sha256: string;
        issues: string[];
      }>(`/api/decision-center/jobs/${selected!.id}/integrity`),
    enabled: !!selected,
  });
  const readiness = useQuery({
    queryKey: ["decision-readiness", selected?.id],
    queryFn: () => api<{ score: number; ready: boolean; blockers: string[]; checks: { key: string; label: string; ok: boolean; optional?: boolean; detail: string }[]; human_approval_required: boolean }>(`/api/decision-center/jobs/${selected!.id}/readiness`),
    enabled: !!selected,
  });
  const act = useMutation({
    mutationFn: ({
      job,
      action,
      payload = {},
    }: {
      job: DecisionItem;
      action: string;
      payload?: Record<string, string>;
    }) =>
      api<{ item: DecisionItem }>(
        `/api/decision-center/jobs/${job.id}/action`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action, ...payload }),
        },
      ),
    onSuccess: (d, v) => {
      setSelected(d.item);
      if (!["not_applied", "unhide", "unblock_company"].includes(v.action))
        setLastAction({ job: v.job, action: v.action });
      qc.invalidateQueries({ queryKey: ["decision-centre"] });
      qc.invalidateQueries({ queryKey: ["applications"] });
      notify("Decision saved");
    },
    onError: (e) => notify(e.message, "error"),
  });
  const prepare = useMutation({
    mutationFn: (job: DecisionItem) =>
      api(`/api/decision-center/jobs/${job.id}/prepare`, { method: "POST" }),
    onSuccess: () =>
      notify(
        "Safe application preparation started. Submit remains disabled.",
        "info",
      ),
    onError: (e) => notify(e.message, "error"),
  });
  const backup = useMutation({
    mutationFn: () =>
      api<{ path: string }>("/api/decision-center/backup", { method: "POST" }),
    onSuccess: (d) => notify(`Backup created at ${d.path}`),
    onError: (e) => notify(e.message, "error"),
  });
  const openApplication = async (job: DecisionItem) => {
    try {
      await api(`/api/jobright/${job.id}/open?target=application`, {
        method: "POST",
      });
      notify(
        "Application opened. Return here and confirm after submitting.",
        "info",
      );
    } catch (e) {
      notify(
        e instanceof Error ? e.message : "Could not open application",
        "error",
      );
    }
  };
  const saveView = async () => {
    const name = prompt("Name this saved view");
    if (!name) return;
    await api("/api/decision-center/saved-views", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, filters: { view, query } }),
    });
    qc.invalidateQueries({ queryKey: ["decision-centre"] });
    notify("View saved");
  };
  const undo = () => {
    if (!lastAction) return;
    const action =
      lastAction.action === "never_company"
        ? "unblock_company"
        : lastAction.action === "applied"
          ? "not_applied"
          : "unhide";
    act.mutate({ job: lastAction.job, action });
    setLastAction(null);
  };
  const confirmSubmitted = () => {
    if (!confirming) return;
    act.mutate({
      job: confirming,
      action: "applied",
      payload: {
        requisition_id: requisition,
        submitted_at: new Date().toISOString(),
      },
    });
    setConfirming(null);
    setRequisition("");
  };
  const eligibilityLabel = (job: DecisionItem) =>
    job.eligibility.status === "eligible"
      ? "Verified"
      : job.eligibility.status === "not_eligible"
        ? "Not eligible"
        : "Check JD";
  const counts = decisions.data?.counts || {};
  const items = decisions.data?.items || [];
  const compared = items.filter((x) => compare.includes(x.id));
  return (
    <div className="page">
      <PageTitle
        eyebrow="Decision intelligence"
        title="Know why every job is shown or hidden."
        description="Review matching evidence, eligibility, documents and agent preparation before taking action."
        action={
          <>
            <button className="button ghost" onClick={saveView}>
              Save view
            </button>
            <a className="button ghost" href="/api/decision-center/export">
              Export workspace
            </a>
            <button
              className="button primary"
              disabled={backup.isPending}
              onClick={() => backup.mutate()}
            >
              {backup.isPending ? "Backing up…" : "Create backup"}
            </button>
          </>
        }
      />
      <div className="metrics decision-metrics">
        {[
          ["Applications", analytics.data?.stats.total],
          ["Actionable", counts.actionable],
          ["Review needed", counts.review],
          ["Already applied", counts.applied],
          ["Ineligible", counts.ineligible],
        ].map(([label, value]) => (
          <div className="metric" key={String(label)}>
            <span>{label}</span>
            <strong>
              {typeof value === "number" ? value.toLocaleString() : "—"}
            </strong>
          </div>
        ))}
      </div>
      {lastAction ? (
        <div className="decision-undo">
          <span>
            Decision updated for <b>{lastAction.job.company}</b>.
          </span>
          <button onClick={undo}>Undo</button>
          <button aria-label="Dismiss" onClick={() => setLastAction(null)}>
            <X size={14} />
          </button>
        </div>
      ) : null}
      <div className="decision-tabs">
        {[
          ["actionable", "Actionable"],
          ["review", "Review needed"],
          ["applied", "Applied"],
          ["ineligible", "Ineligible"],
          ["hidden", "Hidden"],
          ["all", "All jobs"],
        ].map(([id, label]) => (
          <button
            key={id}
            className={view === id ? "active" : ""}
            onClick={() => {
              setView(id);
              setSelected(null);
            }}
          >
            {label}
            <span>
              {id === "all"
                ? Object.values(counts).reduce((a, b) => a + b, 0)
                : counts[id] || 0}
            </span>
          </button>
        ))}
      </div>
      <div className={`decision-grid${selected ? " has-selection" : ""}`}>
        <section className="panel decision-list">
          <div className="toolbar">
            <label className="search">
              <Search size={17} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search company, role, or location"
              />
            </label>
            <span className="feed-count">
              {decisions.data?.total || 0} unique jobs
            </span>
          </div>
          {decisions.isLoading ? (
            <Skeleton rows={9} />
          ) : items.length ? (
            <div>
              {items.map((job) => (
                <button
                  key={job.id}
                  className={selected?.id === job.id ? "selected" : ""}
                  onClick={() => setSelected(job)}
                >
                  <input
                    type="checkbox"
                    checked={compare.includes(job.id)}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) =>
                      setCompare((v) =>
                        e.target.checked
                          ? [...v.slice(-2), job.id]
                          : v.filter((x) => x !== job.id),
                      )
                    }
                  />
                  <span className="company-mark">
                    {job.company.slice(0, 2).toUpperCase()}
                  </span>
                  <span>
                    <b>{job.title}</b>
                    <small>
                      {job.company}
                      {job.location ? ` · ${job.location}` : ""}
                      {(job.duplicate_count || 1) > 1
                        ? ` · ${job.duplicate_count} duplicate captures`
                        : ""}
                    </small>
                    <time>
                      {job.posted || fmt(job.captured_at)} ·{" "}
                      {job.decision.replace("_", " ")}
                    </time>
                  </span>
                  <em
                    className={`eligibility-label eligibility-${job.eligibility.status}`}
                  >
                    {eligibilityLabel(job)}
                  </em>
                </button>
              ))}
              {items.length < (decisions.data?.total || 0) ? (
                <button
                  className="decision-load-more"
                  onClick={() => setLimit((x) => x + 80)}
                >
                  Load 80 more jobs
                </button>
              ) : null}
            </div>
          ) : (
            <Empty
              title="No jobs in this view"
              copy="No records match this decision and search. Choose another view or clear the search."
            />
          )}
        </section>
        <aside className="panel decision-detail">
          {selected ? (
            <>
              <header>
                <button
                  className="decision-back"
                  onClick={() => setSelected(null)}
                >
                  <ChevronLeft size={15} />
                  All jobs
                </button>
                <span
                  className={`decision-badge decision-${selected.decision}`}
                >
                  {selected.decision.replace("_", " ")}
                </span>
                <h2>{selected.title}</h2>
                <p>
                  {selected.company}
                  {selected.location ? ` · ${selected.location}` : ""}
                </p>
              </header>
              <div className="decision-body">
                <section>
                  <h3>Why ASTRA made this decision</h3>
                  <p>{selected.decision_reason}</p>
                </section>
                <section>
                  <h3>Eligibility</h3>
                  <strong>{eligibilityLabel(selected)}</strong>
                  <span className="confidence-note">
                    Eligibility confidence {selected.eligibility.score}%
                  </span>
                  {selected.eligibility.reasons.map((x, i) => (
                    <p key={i}>{x}</p>
                  ))}
                  {selected.eligibility.evidence.map((x, i) => (
                    <blockquote key={i}>{x}</blockquote>
                  ))}
                </section>
                <section>
                  <h3>Résumé/application match</h3>
                  {selected.match_details ? (
                    <>
                      <div className="score-row">
                        <span>Company match</span>
                        <b>{selected.match_details.company_similarity}%</b>
                        <span>Role match</span>
                        <b>{selected.match_details.role_similarity}%</b>
                        <span>JD match</span>
                        <b>{selected.match_details.jd_similarity}%</b>
                      </div>
                      <p>
                        {selected.match_details.company} ·{" "}
                        {selected.match_details.role}
                      </p>
                      <small>
                        {selected.match_details.pdf_path || "No PDF attached"}
                      </small>
                    </>
                  ) : (
                    <p>No matching résumé or application was found.</p>
                  )}
                </section>
                <section>
                  <h3>Document integrity</h3>
                  {integrity.isLoading ? (
                    <p>Checking PDF…</p>
                  ) : (
                    <p>
                      {integrity.data?.valid_pdf
                        ? `Valid PDF · ${Math.round((integrity.data.size || 0) / 1024)} KB`
                        : (integrity.data?.issues || ["No matching PDF"])[0]}
                    </p>
                  )}
                </section>
                <section className="readiness-card">
                  <h3>Application readiness · {readiness.data?.score ?? "—"}%</h3>
                  {readiness.isLoading ? <p>Validating application package…</p> : readiness.data?.checks.map((check) => <p key={check.key}><b>{check.ok ? "✓" : check.optional ? "○" : "!"} {check.label}</b><small>{check.detail}</small></p>)}
                  <small>Human approval is always required. ASTRA cannot submit from this screen.</small>
                </section>
                {selected.timeline?.length ? (
                  <section>
                    <h3>Timeline</h3>
                    {selected.timeline.slice(0, 6).map((x, i) => (
                      <p key={i}>
                        {fmt(x.at)} · {x.action.replaceAll("_", " ")}
                        {x.note ? ` · ${x.note}` : ""}
                      </p>
                    ))}
                  </section>
                ) : null}
              </div>
              <footer>
                <details className="decision-more">
                  <summary className="button ghost">
                    <MoreHorizontal size={15} />
                    More
                  </summary>
                  <div>
                    <button
                      onClick={() =>
                        act.mutate({ job: selected, action: "not_applied" })
                      }
                    >
                      Mark not applied
                    </button>
                    <button
                      onClick={() =>
                        act.mutate({ job: selected, action: "hide" })
                      }
                    >
                      Hide this job
                    </button>
                    <button
                      className="danger"
                      onClick={() =>
                        act.mutate({ job: selected, action: "never_company" })
                      }
                    >
                      Never show {selected.company}
                    </button>
                  </div>
                </details>
                <a
                  className="button ghost"
                  href={`/api/decision-center/jobs/${selected.id}/package`}
                >
                  <Download size={14} />
                  Package
                </a>
                <button
                  className="button ghost"
                  disabled={prepare.isPending}
                  onClick={() => prepare.mutate(selected)}
                >
                  Prepare safely
                </button>
                <button
                  className="button ghost"
                  disabled={!selected.application_url}
                  onClick={() => openApplication(selected)}
                >
                  Open application
                </button>
                <button
                  className="button primary"
                  onClick={() => setConfirming(selected)}
                >
                  Confirm submitted
                </button>
              </footer>
            </>
          ) : (
            <Empty
              title="Select a job decision"
              copy="Choose a job to inspect its matching evidence, eligibility and documents."
            />
          )}
        </aside>
      </div>
      {compared.length > 1 ? (
        <section className="panel comparison">
          <div className="section-head">
            <div>
              <h2>Job comparison</h2>
              <p>Compare up to three selected roles.</p>
            </div>
            <button onClick={() => setCompare([])}>Clear</button>
          </div>
          <div>
            {compared.map((x) => (
              <article key={x.id}>
                <b>{x.company}</b>
                <h3>{x.title}</h3>
                <p>
                  {x.eligibility.status.replace("_", " ")} ·{" "}
                  {x.eligibility.score}%
                </p>
                <small>
                  {x.location || "Location not listed"} ·{" "}
                  {x.match || "No match score"}
                </small>
              </article>
            ))}
          </div>
        </section>
      ) : null}
      <Dialog.Root
        open={!!confirming}
        onOpenChange={(open) => {
          if (!open) setConfirming(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content
            className="dialog-content decision-confirm"
            aria-describedby="decision-confirm-copy"
          >
            <div className="dialog-head">
              <div>
                <span className="eyebrow">Application confirmation</span>
                <Dialog.Title>
                  Confirm this application was submitted
                </Dialog.Title>
                <Dialog.Description id="decision-confirm-copy">
                  ASTRA will hide future copies of this exact role and add it to
                  Applications.
                </Dialog.Description>
              </div>
              <button
                className="dialog-close"
                onClick={() => setConfirming(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="dialog-form">
              <label className="field field-wide">
                <span>Company and role</span>
                <input
                  readOnly
                  value={
                    confirming
                      ? `${confirming.company} · ${confirming.title}`
                      : ""
                  }
                />
              </label>
              <label className="field field-wide">
                <span>
                  Requisition ID{" "}
                  <small>Optional, but improves duplicate detection</small>
                </span>
                <input
                  autoFocus
                  value={requisition}
                  onChange={(e) => setRequisition(e.target.value)}
                  placeholder="Example: R-10482"
                />
              </label>
              <div className="confirmation-note field-wide">
                Submission time will be recorded as now. The original job URL
                and selected document match will be preserved.
              </div>
            </div>
            <div className="dialog-foot">
              <button
                className="button ghost"
                onClick={() => setConfirming(null)}
              >
                Cancel
              </button>
              <button
                className="button primary"
                disabled={act.isPending}
                onClick={confirmSubmitted}
              >
                {act.isPending ? "Saving…" : "Confirm submitted"}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}

type ReviewItem = {
  job_id: string;
  company?: string;
  title?: string;
  status: string;
  message?: string;
  updated_at?: string;
  completed_fields?: string[];
  unanswered_fields?: string[];
  answers?: Record<string, string>;
  evidence_path?: string;
};
type BackupItem = { id: string; path: string; files: string[] };
type IntegrityAudit = {
  checked: number;
  healthy: number;
  missing: { application_id: string; path: string }[];
  invalid: { application_id: string; path: string; error: string }[];
  misnamed: unknown[];
  duplicates: unknown[];
};

function AgentOperations() {
  const qc = useQueryClient();
  const [section, setSection] = useState("review");
  const [selected, setSelected] = useState<ReviewItem | null>(null);
  const [profile, setProfile] = useState<Record<string, unknown>>({});
  const [draft, setDraft] = useState("");
  const review = useQuery({
    queryKey: ["agent-review"],
    queryFn: () =>
      api<{ items: ReviewItem[]; total: number }>(
        "/api/decision-center/review",
      ),
  });
  const operations = useQuery({
    queryKey: ["operations"],
    queryFn: () =>
      api<{
        database: {
          path: string;
          schema_version: number;
          checkpoints: number;
          activity_events: number;
          snapshots: number;
        };
        activity: {
          id: number;
          kind: string;
          status: string;
          message: string;
          created_at: string;
        }[];
      }>("/api/decision-center/operations"),
  });
  const integrity = useQuery({
    queryKey: ["integrity-audit"],
    queryFn: () => api<IntegrityAudit>("/api/decision-center/integrity"),
    enabled: section === "integrity",
  });
  const backups = useQuery({
    queryKey: ["backups"],
    queryFn: () =>
      api<{ items: BackupItem[]; total: number }>(
        "/api/decision-center/backups",
      ),
    enabled: section === "recovery",
  });
  const profileQuery = useQuery({
    queryKey: ["eligibility-profile"],
    queryFn: () =>
      api<{ profile: Record<string, unknown> }>("/api/decision-center/profile"),
  });
  useEffect(() => {
    if (profileQuery.data && !Object.keys(profile).length)
      setProfile(profileQuery.data.profile);
  }, [profileQuery.data]);
  const reviewAction = useMutation({
    mutationFn: ({ jobId, action }: { jobId: string; action: string }) =>
      api(`/api/decision-center/review/${jobId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent-review"] });
      notify("Review decision saved");
    },
  });
  const resumeAgent = useMutation({
    mutationFn: (jobId: string) =>
      api(`/api/decision-center/review/${jobId}/resume`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent-review"] });
      notify("Application preparation resumed from its checkpoint", "info");
    },
    onError: (error) => notify(error.message, "error"),
  });
  const saveProfile = useMutation({
    mutationFn: () =>
      api("/api/decision-center/profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["eligibility-profile"] });
      notify("Eligibility profile saved");
    },
  });
  const repair = useMutation({
    mutationFn: (action: string) =>
      api<{ removed?: number }>("/api/decision-center/integrity/repair", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["integrity-audit"] });
      notify("Document library repaired");
    },
  });
  const restore = useMutation({
    mutationFn: (id: string) =>
      api(`/api/decision-center/backups/${id}/restore`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries();
      notify("Backup restored. Current state was preserved first.");
    },
  });
  const createBackup = useMutation({
    mutationFn: () => api("/api/decision-center/backup", { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backups"] });
      notify("Recovery point created");
    },
  });
  const generate = useMutation({
    mutationFn: ({
      jobId,
      kind,
    }: {
      jobId: string;
      kind: "follow-up" | "interview-kit";
    }) =>
      api<Record<string, unknown>>(
        `/api/decision-center/jobs/${jobId}/${kind}`,
        { method: "POST" },
      ),
    onSuccess: (d) =>
      setDraft(
        typeof d.text === "string" ? d.text : JSON.stringify(d, null, 2),
      ),
    onError: (e) => notify(e.message, "error"),
  });
  const set = (key: string, value: string) =>
    setProfile((v) => ({ ...v, [key]: value }));
  const fields: [[string, string, string?], [string, string, string?]][] = [
    [
      ["first_name", "First name"],
      ["last_name", "Last name"],
    ],
    [
      ["email", "Email", "email"],
      ["phone", "Phone"],
    ],
    [
      ["location", "Current address"],
      ["citizenship", "Citizenship"],
    ],
    [
      ["years_experience", "Professional experience"],
      ["immigration_status", "Immigration status"],
    ],
    [
      ["opt_start", "OPT start", "date"],
      ["opt_expiration", "OPT end", "date"],
    ],
    [
      ["stem_opt_start", "STEM OPT start", "date"],
      ["stem_opt_end", "STEM OPT end", "date"],
    ],
    [
      ["available_start", "Available start"],
      ["salary_strategy", "Salary strategy"],
    ],
  ];
  return (
    <div className="page">
      <PageTitle
        eyebrow="Application operations"
        title="Review, recover and repair from one place."
        description="Nothing submits automatically. Inspect agent work, maintain your eligibility profile, repair documents and restore local data."
        action={
          <button
            className="button primary"
            onClick={() => createBackup.mutate()}
            disabled={createBackup.isPending}
          >
            Create recovery point
          </button>
        }
      />
      <div className="decision-tabs">
        {[
          ["review", "Review queue"],
          ["profile", "Eligibility profile"],
          ["integrity", "Document health"],
          ["recovery", "Recovery"],
          ["activity", "Activity"],
        ].map(([id, label]) => (
          <button
            className={section === id ? "active" : ""}
            onClick={() => setSection(id)}
            key={id}
          >
            {label}
            {id === "review" ? <span>{review.data?.total || 0}</span> : null}
          </button>
        ))}
      </div>
      {section === "review" ? (
        <div className="operations-grid">
          <section className="panel operation-list">
            {review.isLoading ? (
              <Skeleton rows={7} />
            ) : review.data?.items.length ? (
              review.data.items.map((item) => (
                <button
                  className={selected?.job_id === item.job_id ? "selected" : ""}
                  key={`${item.job_id}-${item.updated_at || item.status}`}
                  onClick={() => {
                    setSelected(item);
                    setDraft("");
                  }}
                >
                  <span className={`agent-status agent-status-${item.status}`}>
                    {item.status.replaceAll("_", " ")}
                  </span>
                  <b>{item.company || "Unknown company"}</b>
                  <small>{item.title || item.job_id}</small>
                  <time>{fmt(item.updated_at)}</time>
                </button>
              ))
            ) : (
              <Empty
                title="Review queue is clear"
                copy="Prepared applications and unresolved fields will appear here."
              />
            )}
          </section>
          <section className="panel operation-detail">
            {selected ? (
              <>
                <header>
                  <span className="eyebrow">Human approval required</span>
                  <h2>{selected.company}</h2>
                  <p>{selected.title}</p>
                </header>
                <div className="operation-body">
                  <h3>Agent result</h3>
                  <p>{selected.message || "No message recorded."}</p>
                  <h3>Completed fields</h3>
                  <div className="tag-list">
                    {selected.completed_fields?.length ? (
                      selected.completed_fields.map((x) => (
                        <span key={x}>{x}</span>
                      ))
                    ) : (
                      <p>No completed-field evidence recorded.</p>
                    )}
                  </div>
                  <h3>Needs your answer</h3>
                  {selected.unanswered_fields?.length ? (
                    selected.unanswered_fields.map((x) => (
                      <blockquote key={x}>{x}</blockquote>
                    ))
                  ) : (
                    <p>All required fields were completed.</p>
                  )}
                  {Object.keys(selected.answers || {}).length ? (
                    <>
                      <h3>Generated answers</h3>
                      {Object.entries(selected.answers || {}).map(([q, a]) => (
                        <div className="answer-proof" key={q}>
                          <b>{q}</b>
                          <p>{a}</p>
                        </div>
                      ))}
                    </>
                  ) : null}
                  {draft ? (
                    <pre className="generated-draft">{draft}</pre>
                  ) : null}
                </div>
                <footer>
                  <button
                    className="button ghost"
                    onClick={() =>
                      generate.mutate({
                        jobId: selected.job_id,
                        kind: "follow-up",
                      })
                    }
                  >
                    Draft follow-up
                  </button>
                  <button
                    className="button ghost"
                    onClick={() =>
                      generate.mutate({
                        jobId: selected.job_id,
                        kind: "interview-kit",
                      })
                    }
                  >
                    Interview kit
                  </button>
                  <button
                    className="button ghost"
                    onClick={() =>
                      reviewAction.mutate({
                        jobId: selected.job_id,
                        action: "review_rejected",
                      })
                    }
                  >
                    Reject
                  </button>
                  <button
                    className="button primary"
                    onClick={() => {
                      reviewAction.mutate({
                        jobId: selected.job_id,
                        action: "review_approved",
                      });
                      resumeAgent.mutate(selected.job_id);
                    }}
                  >
                    Approve and resume
                  </button>
                </footer>
              </>
            ) : (
              <Empty
                title="Select an agent run"
                copy="Review every populated field and unresolved question before approval."
              />
            )}
          </section>
        </div>
      ) : null}
      {section === "profile" ? (
        <section className="panel padded profile-centre">
          <div className="section-head">
            <div>
              <h2>Eligibility and application defaults</h2>
              <p>
                These values drive eligibility decisions and factual form
                answers.
              </p>
            </div>
            <button
              className="button primary"
              onClick={() => saveProfile.mutate()}
              disabled={saveProfile.isPending}
            >
              Save profile
            </button>
          </div>
          <div className="profile-fields">
            {fields.flat().map(([key, label, type]) => (
              <label className="field" key={key}>
                <span>{label}</span>
                <input
                  type={type || "text"}
                  value={String(profile[key] || "")}
                  onChange={(e) => set(key, e.target.value)}
                />
              </label>
            ))}
            {[
              ["authorized_to_work", "Authorized to work in US"],
              ["requires_sponsorship", "Requires sponsorship"],
              ["currently_student", "Currently a student"],
              ["masters_completed", "Master’s completed"],
              ["security_clearance", "US security clearance"],
            ].map(([key, label]) => (
              <label className="field" key={key}>
                <span>{label}</span>
                <select
                  value={String(profile[key] || "")}
                  onChange={(e) => set(key, e.target.value)}
                >
                  <option value="">Choose…</option>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </label>
            ))}
          </div>
        </section>
      ) : null}
      {section === "integrity" ? (
        <section className="panel padded">
          <div className="section-head">
            <div>
              <h2>Document integrity</h2>
              <p>Validate every attached PDF before using it.</p>
            </div>
            <div>
              <button
                className="button ghost"
                onClick={() => repair.mutate("rescan")}
              >
                Rescan library
              </button>{" "}
              <button
                className="button primary"
                onClick={() => repair.mutate("forget_missing")}
                disabled={!integrity.data?.missing.length}
              >
                Remove missing references
              </button>
            </div>
          </div>
          {integrity.isLoading ? (
            <Skeleton rows={5} />
          ) : integrity.data ? (
            <>
              <div className="metrics">
                {[
                  ["Checked", integrity.data.checked],
                  ["Healthy", integrity.data.healthy],
                  ["Missing", integrity.data.missing.length],
                  ["Invalid", integrity.data.invalid.length],
                ].map(([a, b]) => (
                  <div className="metric" key={String(a)}>
                    <span>{a}</span>
                    <strong>{b}</strong>
                  </div>
                ))}
              </div>
              <div className="issue-list">
                {[
                  ...integrity.data.missing.map((x) => ({
                    ...x,
                    error: "File is missing",
                  })),
                  ...integrity.data.invalid,
                ].map((x, i) => (
                  <div key={`${x.path}-${i}`}>
                    <b>{x.error}</b>
                    <code>{x.path}</code>
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </section>
      ) : null}
      {section === "recovery" ? (
        <section className="panel padded">
          <div className="section-head">
            <div>
              <h2>Recovery points</h2>
              <p>Restoring first creates a safety copy of the current state.</p>
            </div>
          </div>
          {backups.isLoading ? (
            <Skeleton rows={5} />
          ) : (
            <div className="backup-list">
              {backups.data?.items.map((x) => (
                <div key={x.id}>
                  <span>
                    <b>{x.id}</b>
                    <small>{x.files.join(" · ")}</small>
                  </span>
                  <button
                    className="button ghost"
                    disabled={restore.isPending}
                    onClick={() => {
                      if (
                        confirm(
                          `Restore ${x.id}? ASTRA will preserve the current state first.`,
                        )
                      )
                        restore.mutate(x.id);
                    }}
                  >
                    Restore
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      ) : null}
      {section === "activity" ? (
        <section className="panel padded">
          <div className="section-head">
            <div>
              <h2>System activity</h2>
              <p>Persistent SQLite audit history and workflow checkpoints.</p>
            </div>
            <span className="secure">
              <Database size={13} /> Schema{" "}
              {operations.data?.database.schema_version || "—"}
            </span>
          </div>
          <div className="activity-list">
            {operations.data?.activity.map((x) => (
              <div key={x.id}>
                <span className={`agent-status agent-status-${x.status}`}>
                  {x.status}
                </span>
                <p>
                  <b>{x.kind}</b>
                  <small>{x.message}</small>
                </p>
                <time>{fmt(x.created_at)}</time>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function Applications() {
  const [focus] = useState<{ id?: string; company?: string }>(() => {
    try {
      return JSON.parse(
        localStorage.getItem("astra-focus-application") || "{}",
      );
    } catch {
      return {};
    }
  });
  const [query, setQuery] = useState(focus.company || "");
  const [status, setStatus] = useState("all");
  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState<Application | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [display, setDisplay] = useState<"table" | "board">(() =>
    localStorage.getItem("astra-app-view") === "board" ? "board" : "table",
  );
  const qc = useQueryClient();
  const limit = 50;
  useEffect(() => localStorage.setItem("astra-app-view", display), [display]);
  const result = useQuery({
    queryKey: ["applications", query, status, page],
    queryFn: () =>
      api<ApplicationsResponse>(
        `/api/applications?limit=${limit}&page=${page}&q=${encodeURIComponent(query)}&status=${status}`,
      ),
    placeholderData: (previous) => previous,
  });
  useEffect(() => {
    if (focus.id && !editing && result.data?.items.length) {
      const found = result.data.items.find((x) => x.id === focus.id);
      if (found) {
        setEditing(found);
        localStorage.removeItem("astra-focus-application");
      }
    }
  }, [focus.id, editing, result.data]);
  const scan = useMutation({
    mutationFn: () =>
      api<{ total_applications: number }>("/api/applications/scan", {
        method: "POST",
      }),
    onSuccess: (d) => {
      setPage(1);
      qc.invalidateQueries({ queryKey: ["applications"] });
      notify(`${d.total_applications.toLocaleString()} applications indexed`);
    },
    onError: (e) => notify(e.message, "error"),
  });
  const save = useMutation({
    mutationFn: (payload: Record<string, string | null>) =>
      api<Application>(`/api/applications/${editing!.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    onSuccess: (d) => {
      setEditing(null);
      qc.invalidateQueries({ queryKey: ["applications"] });
      notify(`${d.company} moved to ${d.status.replace("_", " ")}`);
    },
    onError: (e) => notify(e.message, "error"),
  });
  const bulk = useMutation({
    mutationFn: (nextStatus: string) =>
      api<{ updated: number }>("/api/applications/bulk", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ids: selectedIds,
          patch: { status: nextStatus },
        }),
      }),
    onSuccess: (d) => {
      setSelectedIds([]);
      qc.invalidateQueries({ queryKey: ["applications"] });
      qc.invalidateQueries({ queryKey: ["application-archive"] });
      notify(`${d.updated} applications updated`);
    },
    onError: (e) => notify(e.message, "error"),
  });
  const data = result.data;
  return (
    <div className="page">
      <PageTitle
        eyebrow="Application intelligence"
        title="Your search, clearly organised."
        description="Every opportunity, follow-up and outcome in one focused workspace."
        action={
          <>
            <button className="button ghost" onClick={() => result.refetch()}>
              <RefreshCw size={16} /> Refresh
            </button>
            <button
              className="button primary"
              onClick={() => scan.mutate()}
              disabled={scan.isPending}
            >
              <Sparkles size={16} />
              {scan.isPending ? "Scanning…" : "Scan archive"}
            </button>
          </>
        }
      />
      <div className="metrics">
        {[
          ["Applications", data?.stats.total],
          ["Companies", data?.stats.companies],
          ["Recruiter calls", data?.stats.calls],
          ["Interviews", data?.stats.interviews],
          ["Offers", data?.stats.offers],
        ].map(([label, value]) => (
          <div className="metric" key={String(label)}>
            <span>{label}</span>
            <strong>
              {typeof value === "number" ? value.toLocaleString() : "—"}
            </strong>
          </div>
        ))}
      </div>
      <section className="panel">
        <div className="toolbar">
          <label className="search">
            <Search size={17} />
            <input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(1);
              }}
              placeholder="Search company or role"
            />
          </label>
          <select
            value={status}
            onChange={(e) => {
              setStatus(e.target.value);
              setPage(1);
            }}
          >
            <option value="all">All stages</option>
            <option value="saved">Saved</option>
            <option value="applied">Applied</option>
            <option value="recruiter_call">Recruiter call</option>
            <option value="interview">Interview</option>
            <option value="offer">Offer</option>
            <option value="rejected">Rejected</option>
            <option value="withdrawn">Withdrawn</option>
          </select>
          <div className="segmented view-switch">
            <button
              className={display === "table" ? "active" : ""}
              onClick={() => setDisplay("table")}
            >
              Table
            </button>
            <button
              className={display === "board" ? "active" : ""}
              onClick={() => setDisplay("board")}
            >
              Board
            </button>
          </div>
        </div>
        {selectedIds.length > 0 && (
          <div className="bulk-bar">
            <b>{selectedIds.length} selected</b>
            <select
              defaultValue=""
              onChange={(e) => {
                if (e.target.value) bulk.mutate(e.target.value);
              }}
            >
              <option value="" disabled>
                Move to stage…
              </option>
              <option value="applied">Applied</option>
              <option value="recruiter_call">Recruiter call</option>
              <option value="interview">Interview</option>
              <option value="offer">Offer</option>
              <option value="rejected">Rejected</option>
            </select>
            <button onClick={() => setSelectedIds([])}>Clear</button>
          </div>
        )}
        {result.isLoading ? (
          <Skeleton rows={8} />
        ) : result.isError ? (
          <Empty
            title="Applications unavailable"
            copy="Check that the ASTRA service is running, then refresh."
            action={
              <button className="button ghost" onClick={() => result.refetch()}>
                Try again
              </button>
            }
          />
        ) : !data?.items.length ? (
          <Empty
            title="No matching applications"
            copy="Try a different search or scan your ASTRA archive."
          />
        ) : display === "board" ? (
          <>
            <div className="app-board">
              {["saved", "applied", "recruiter_call", "interview", "offer"].map(
                (stage) => (
                  <section key={stage}>
                    <header>
                      <b>{stage.replace("_", " ")}</b>
                      <span>
                        {data.items.filter((x) => x.status === stage).length}
                      </span>
                    </header>
                    {data.items
                      .filter((x) => x.status === stage)
                      .map((a) => (
                        <button key={a.id} onClick={() => setEditing(a)}>
                          <strong>{a.company}</strong>
                          <small>{a.role}</small>
                          <time>{fmt(a.applied_at)}</time>
                        </button>
                      ))}
                  </section>
                ),
              )}
            </div>
            <div className="pagination">
              <span>Showing page {page} of the pipeline</span>
              <div>
                <button
                  className="button ghost"
                  disabled={page === 1}
                  onClick={() => setPage((v) => v - 1)}
                >
                  Previous
                </button>
                <button
                  className="button ghost"
                  disabled={page * limit >= data.total}
                  onClick={() => setPage((v) => v + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        ) : (
          <>
            <div
              className={`table-wrap ${result.isFetching ? "table-refreshing" : ""}`}
            >
              <table>
                <thead>
                  <tr>
                    <th>
                      <input
                        type="checkbox"
                        aria-label="Select page"
                        checked={
                          !!data.items.length &&
                          data.items.every((x) => selectedIds.includes(x.id))
                        }
                        onChange={(e) =>
                          setSelectedIds(
                            e.target.checked ? data.items.map((x) => x.id) : [],
                          )
                        }
                      />
                    </th>
                    <th>Company and role</th>
                    <th>Stage</th>
                    <th>Applied</th>
                    <th>Recruiter call</th>
                    <th>Files</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((a) => (
                    <tr
                      key={a.id}
                      className={
                        selectedIds.includes(a.id) ? "row-selected" : ""
                      }
                    >
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedIds.includes(a.id)}
                          onChange={(e) =>
                            setSelectedIds((v) =>
                              e.target.checked
                                ? [...v, a.id]
                                : v.filter((id) => id !== a.id),
                            )
                          }
                          aria-label={`Select ${a.company}`}
                        />
                      </td>
                      <td>
                        <b>{a.company}</b>
                        <span>{a.role}</span>
                      </td>
                      <td>
                        <em className={`status status-${a.status}`}>
                          {a.status.replace("_", " ")}
                        </em>
                      </td>
                      <td>{fmt(a.applied_at)}</td>
                      <td>{fmt(a.call_at)}</td>
                      <td>{a.files?.length || 0}</td>
                      <td>
                        <div className="row-actions">
                          {a.job_url ? (
                            <a
                              className="row-edit"
                              href={a.job_url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              <ExternalLink size={14} />
                              Job
                            </a>
                          ) : null}
                          {a.files?.length ? (
                            <button
                              className="row-edit"
                              onClick={() => {
                                localStorage.setItem(
                                  "astra-preview-context",
                                  `app:${a.id}`,
                                );
                                location.hash = "#/preview";
                              }}
                            >
                              <FileCheck2 size={14} />
                              Résumé
                            </button>
                          ) : null}
                          <button
                            className="row-edit"
                            onClick={() => setEditing(a)}
                          >
                            <Edit3 size={14} />
                            Update
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span>
                {((page - 1) * limit + 1).toLocaleString()}–
                {Math.min(page * limit, data.total).toLocaleString()} of{" "}
                {data.total.toLocaleString()}
              </span>
              <div>
                <button
                  className="button ghost"
                  disabled={page === 1 || result.isFetching}
                  onClick={() => setPage((v) => Math.max(1, v - 1))}
                >
                  Previous
                </button>
                <span>
                  Page {page} of {Math.max(1, Math.ceil(data.total / limit))}
                </span>
                <button
                  className="button ghost"
                  disabled={page * limit >= data.total || result.isFetching}
                  onClick={() => setPage((v) => v + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </section>
      <ApplicationEditor
        application={editing}
        saving={save.isPending}
        error={save.isError ? save.error.message : ""}
        onClose={() => setEditing(null)}
        onSave={(payload) => save.mutate(payload)}
      />
    </div>
  );
}

function ApplicationEditor({
  application,
  saving,
  error,
  onClose,
  onSave,
}: {
  application: Application | null;
  saving: boolean;
  error: string;
  onClose: () => void;
  onSave: (payload: Record<string, string | null>) => void;
}) {
  const [form, setForm] = useState<Record<string, string>>({});
  const [initial, setInitial] = useState("");
  const [eventType, setEventType] = useState("note");
  const [eventNote, setEventNote] = useState("");
  const [eventPerson, setEventPerson] = useState("");
  const qc = useQueryClient();
  const addEvent = useMutation({
    mutationFn: () =>
      api<TimelineEvent>(`/api/applications/${application!.id}/timeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: eventType,
          note: eventNote,
          person: eventPerson,
        }),
      }),
    onSuccess: (event) => {
      if (application)
        application.timeline = [event, ...(application.timeline || [])];
      setEventNote("");
      setEventPerson("");
      qc.invalidateQueries({ queryKey: ["applications"] });
      notify("Timeline event added");
    },
    onError: (e) => notify(e.message, "error"),
  });
  useEffect(() => {
    if (application) {
      const next = {
        status: application.status || "applied",
        applied_at: localDate(application.applied_at),
        call_at: localDate(application.call_at),
        interview_at: localDate(application.interview_at),
        offer_at: localDate(application.offer_at),
        follow_up_at: localDate(application.follow_up_at),
        source: application.source || "",
        job_url: application.job_url || "",
        notes: application.notes || "",
      };
      setForm(next);
      setInitial(JSON.stringify(next));
    }
  }, [application]);
  const set = (key: string, value: string) =>
    setForm((v) => ({ ...v, [key]: value }));
  const setStage = (value: string) =>
    setForm((v) => {
      const next: Record<string, string> = { ...v, status: value },
        now = new Date().toISOString().slice(0, 16);
      if (value === "applied" && !next.applied_at) next.applied_at = now;
      if (value === "recruiter_call" && !next.call_at) next.call_at = now;
      if (value === "interview" && !next.interview_at) next.interview_at = now;
      if (value === "offer" && !next.offer_at) next.offer_at = now;
      return next;
    });
  const requestClose = () => {
    if (
      JSON.stringify(form) !== initial &&
      !confirm("Discard unsaved application changes?")
    )
      return;
    onClose();
  };
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const payload: Record<string, string | null> = { ...form };
    for (const key of [
      "applied_at",
      "call_at",
      "interview_at",
      "offer_at",
      "follow_up_at",
    ])
      payload[key] = form[key] ? new Date(form[key]).toISOString() : null;
    if (form.status === "rejected" && !application?.rejected_at)
      payload.rejected_at = new Date().toISOString();
    onSave(payload);
  };
  return (
    <Dialog.Root
      open={!!application}
      onOpenChange={(open) => {
        if (!open && !saving) requestClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content
          className="dialog-content"
          aria-describedby="application-dialog-description"
          onEscapeKeyDown={(e) => {
            if (JSON.stringify(form) !== initial) {
              e.preventDefault();
              requestClose();
            }
          }}
        >
          <div className="dialog-head">
            <div>
              <span className="eyebrow">Application record</span>
              <Dialog.Title>{application?.company}</Dialog.Title>
              <Dialog.Description id="application-dialog-description">
                {application?.role}
              </Dialog.Description>
            </div>
            <button
              type="button"
              className="dialog-close"
              disabled={saving}
              onClick={requestClose}
            >
              <X size={18} />
            </button>
          </div>
          <form onSubmit={submit}>
            <div className="dialog-form">
              <label className="field">
                <span>Application stage</span>
                <select
                  value={form.status || "applied"}
                  onChange={(e) => setStage(e.target.value)}
                >
                  <option value="saved">Saved</option>
                  <option value="applied">Applied</option>
                  <option value="recruiter_call">Recruiter call</option>
                  <option value="interview">Interview</option>
                  <option value="offer">Offer</option>
                  <option value="rejected">Rejected</option>
                  <option value="withdrawn">Withdrawn</option>
                </select>
              </label>
              <label className="field">
                <span>Source</span>
                <input
                  value={form.source || ""}
                  onChange={(e) => set("source", e.target.value)}
                  placeholder="LinkedIn, referral…"
                />
              </label>
              <label className="field">
                <span>Applied date</span>
                <input
                  type="datetime-local"
                  value={form.applied_at || ""}
                  onChange={(e) => set("applied_at", e.target.value)}
                />
              </label>
              <label className="field">
                <span>Recruiter call</span>
                <input
                  type="datetime-local"
                  value={form.call_at || ""}
                  onChange={(e) => set("call_at", e.target.value)}
                />
              </label>
              <label className="field">
                <span>Interview date</span>
                <input
                  type="datetime-local"
                  value={form.interview_at || ""}
                  onChange={(e) => set("interview_at", e.target.value)}
                />
              </label>
              <label className="field">
                <span>Offer date</span>
                <input
                  type="datetime-local"
                  value={form.offer_at || ""}
                  onChange={(e) => set("offer_at", e.target.value)}
                />
              </label>
              <label className="field">
                <span>Follow-up date</span>
                <input
                  type="datetime-local"
                  value={form.follow_up_at || ""}
                  onChange={(e) => set("follow_up_at", e.target.value)}
                />
              </label>
              <label className="field field-wide">
                <span>Job URL</span>
                <input
                  type="url"
                  value={form.job_url || ""}
                  onChange={(e) => set("job_url", e.target.value)}
                  placeholder="https://…"
                />
              </label>
              <label className="field field-wide">
                <span>Notes</span>
                <textarea
                  value={form.notes || ""}
                  onChange={(e) => set("notes", e.target.value)}
                  placeholder="Recruiter, interview details, next action…"
                />
              </label>
              <div className="timeline-editor field-wide">
                <div className="section-head">
                  <div>
                    <h3>Application timeline</h3>
                    <p>Add interview rounds, assessments, contacts and follow-ups.</p>
                  </div>
                </div>
                <div className="timeline-compose">
                  <select value={eventType} onChange={(e) => setEventType(e.target.value)}>
                    <option value="note">Note</option>
                    <option value="recruiter_contact">Recruiter contact</option>
                    <option value="assessment">Assessment</option>
                    <option value="interview">Interview round</option>
                    <option value="follow_up">Follow-up</option>
                    <option value="offer">Offer</option>
                    <option value="rejected">Rejection</option>
                    <option value="document">Document</option>
                  </select>
                  <input value={eventPerson} onChange={(e) => setEventPerson(e.target.value)} placeholder="Person or interviewer" />
                  <input value={eventNote} onChange={(e) => setEventNote(e.target.value)} placeholder="What happened?" />
                  <button type="button" className="button ghost" disabled={!eventNote.trim() || addEvent.isPending} onClick={() => addEvent.mutate()}>
                    Add event
                  </button>
                </div>
                <div className="timeline-history">
                  {application?.timeline?.length ? application.timeline.map((item) => (
                    <div key={item.id}>
                      <time>{fmt(item.at)}</time>
                      <b>{item.type.replaceAll("_", " ")}</b>
                      <span>{item.person ? `${item.person} · ` : ""}{item.note}</span>
                    </div>
                  )) : <small>No detailed events recorded yet.</small>}
                </div>
              </div>
              {error && <div className="inline-error field-wide">{error}</div>}
            </div>
            <div className="dialog-foot">
              <button
                type="button"
                className="button ghost"
                onClick={requestClose}
                disabled={saving}
              >
                Cancel
              </button>
              {application?.job_url ? (
                <a
                  className="button ghost"
                  href={application.job_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open job
                </a>
              ) : null}
              {application?.files?.length ? (
                <a
                  className="button ghost"
                  href={`/api/applications/${application.id}/resume`}
                  target="_blank"
                >
                  Open résumé
                </a>
              ) : null}
              <button className="button primary" disabled={saving}>
                {saving ? "Saving…" : "Save changes"}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ToastViewport() {
  const [items, setItems] = useState<
    { id: number; message: string; tone: string }[]
  >([]);
  useEffect(() => {
    const handler = (event: Event) => {
      const { message, tone } = (event as CustomEvent).detail,
        id = Date.now();
      setItems((v) => [...v, { id, message, tone }]);
      setTimeout(() => setItems((v) => v.filter((x) => x.id !== id)), 3200);
    };
    addEventListener("astra:toast", handler);
    return () => removeEventListener("astra:toast", handler);
  }, []);
  return (
    <div className="toast-viewport" aria-live="polite">
      {items.map((item) => (
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0 }}
          className={`toast toast-${item.tone}`}
          key={item.id}
        >
          {item.message}
        </motion.div>
      ))}
    </div>
  );
}

function Genie() {
  const result = useQuery({
    queryKey: ["applications", "genie"],
    queryFn: () => api<ApplicationsResponse>("/api/applications?limit=5"),
  });
  const data = result.data;
  const archive = useQuery({
    queryKey: ["application-archive"],
    queryFn: fetchApplicationArchive,
    staleTime: 300000,
  });
  const today = Date.now(),
    day = 86400000;
  const overdue = (archive.data || []).filter(
    (x) =>
      x.follow_up_at &&
      new Date(x.follow_up_at).getTime() < today &&
      !["offer", "rejected", "withdrawn"].includes(x.status),
  );
  const interviews = (archive.data || [])
    .filter(
      (x) =>
        x.interview_at &&
        new Date(x.interview_at).getTime() >= today &&
        new Date(x.interview_at).getTime() <= today + 7 * day,
    )
    .sort((a, b) =>
      String(a.interview_at).localeCompare(String(b.interview_at)),
    );
  const actions = [
    {
      to: "studio",
      icon: Command,
      title: "Tailor a résumé",
      copy: "Create a focused, ATS-ready version for a new role.",
    },
    {
      to: "talk",
      icon: MessageCircleMore,
      title: "Answer an application question",
      copy: "Write a factual answer in your natural voice.",
    },
    {
      to: "applications",
      icon: BriefcaseBusiness,
      title: "Update an application",
      copy: "Record a recruiter call, interview or follow-up.",
    },
    {
      to: "cover",
      icon: FileText,
      title: "Create a cover letter",
      copy: "Build a role-specific letter from saved context.",
    },
  ] as const;
  return (
    <div className="page">
      <PageTitle
        eyebrow="Daily command centre"
        title="Good to see you, Sri."
        description="Move your job search forward without hunting through tools."
      />
      <div className="genie-grid">
        <section className="panel padded span-2">
          <div className="section-head">
            <div>
              <h2>What do you want to do?</h2>
              <p>Start with an outcome.</p>
            </div>
            <span className="live">
              <i />
              Ready
            </span>
          </div>
          <div className="action-grid">
            {actions.map((a) => {
              const Icon = a.icon;
              return (
                <motion.a
                  whileHover={{ y: -3 }}
                  href={`#/${a.to}`}
                  className="action-card"
                  key={a.to}
                >
                  <span className="action-icon">
                    <Icon size={20} />
                  </span>
                  <b>{a.title}</b>
                  <p>{a.copy}</p>
                  <ChevronRight size={17} />
                </motion.a>
              );
            })}
          </div>
        </section>
        <section className="panel padded">
          <div className="section-head">
            <div>
              <h2>Search pulse</h2>
              <p>Live tracker data</p>
            </div>
          </div>
          {result.isLoading ? (
            <Skeleton rows={3} />
          ) : (
            <div className="pulse">
              <div>
                <strong>{data?.stats.total.toLocaleString() || "0"}</strong>
                <span>Applications</span>
              </div>
              <div>
                <strong>{data?.stats.calls.toLocaleString() || "0"}</strong>
                <span>Calls</span>
              </div>
              <div>
                <strong>{data?.stats.response_rate || 0}%</strong>
                <span>Response</span>
              </div>
            </div>
          )}
        </section>
        <section className="panel padded span-3">
          <div className="section-head">
            <div>
              <h2>Application funnel</h2>
              <p>Where your opportunities currently stand.</p>
            </div>
            <a href="#/applications">
              Manage pipeline <ChevronRight size={14} />
            </a>
          </div>
          <div className="funnel">
            {[
              ["Applied", data?.stats.status_counts?.applied || 0],
              ["Recruiter calls", data?.stats.calls || 0],
              ["Interviews", data?.stats.interviews || 0],
              ["Offers", data?.stats.offers || 0],
            ].map(([label, value], i) => (
              <div key={String(label)}>
                <span>
                  <b>{String(label)}</b>
                  <em>{Number(value).toLocaleString()}</em>
                </span>
                <i>
                  <u
                    style={{
                      width: `${data?.stats.total ? Math.max(3, (Number(value) / data.stats.total) * 100) : 0}%`,
                    }}
                  />
                </i>
                {i < 3 && <ChevronRight size={14} />}
              </div>
            ))}
          </div>
        </section>
        <section className="panel padded span-3">
          <div className="section-head">
            <div>
              <h2>Needs your attention</h2>
              <p>Time-sensitive actions from your tracker.</p>
            </div>
          </div>
          <div className="attention-grid">
            <a
              href="#/applications"
              onClick={() =>
                localStorage.setItem("astra-application-preset", "overdue")
              }
            >
              <CalendarClock size={19} />
              <span>
                <strong>{overdue.length}</strong>
                <b>Overdue follow-ups</b>
                <small>
                  {overdue.length
                    ? `${overdue[0].company} is waiting for an update.`
                    : "Nothing overdue."}
                </small>
              </span>
              <ChevronRight size={16} />
            </a>
            <a href="#/applications">
              <BriefcaseBusiness size={19} />
              <span>
                <strong>{interviews.length}</strong>
                <b>Interviews this week</b>
                <small>
                  {interviews.length
                    ? `${interviews[0].company} · ${fmt(interviews[0].interview_at)}`
                    : "No interviews scheduled."}
                </small>
              </span>
              <ChevronRight size={16} />
            </a>
          </div>
        </section>
        <section className="panel padded span-3">
          <div className="section-head">
            <div>
              <h2>Recent applications</h2>
              <p>Continue where you left off.</p>
            </div>
            <a href="#/applications">
              View all <ChevronRight size={14} />
            </a>
          </div>
          {result.isLoading ? (
            <Skeleton />
          ) : (
            <div className="recent-list">
              {data?.items.map((a) => (
                <div className="recent-row" key={a.id}>
                  <div className="company-mark">
                    {a.company.slice(0, 2).toUpperCase()}
                  </div>
                  <div>
                    <b>{a.company}</b>
                    <span>{a.role}</span>
                  </div>
                  <em className={`status status-${a.status}`}>
                    {a.status.replace("_", " ")}
                  </em>
                  <time>{fmt(a.applied_at)}</time>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function Talk() {
  const contexts = useQuery({
    queryKey: ["contexts"],
    queryFn: () => api<{ items: Context[] }>("/api/context/list?limit=5000"),
  });
  const [key, setKey] = useState("");
  const [tone, setTone] = useState("balanced");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const selected = contexts.data?.items.find((x) => x.key === key);
  const generate = useMutation({
    mutationFn: () =>
      api<{ final_text?: string; answer?: string }>("/api/talk/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          context_key: key,
          use_latest: false,
          tone,
          humanize: true,
          include_followups: false,
        }),
      }),
    onSuccess: (d) => setAnswer(d.final_text || d.answer || ""),
  });
  const presets = [
    "Why are you interested in this role?",
    "Why do you want to work at this company?",
    "Describe your relevant experience.",
    "What makes you a good fit?",
  ];
  return (
    <div className="page">
      <PageTitle
        eyebrow="Context-aware writing"
        title="A clear answer, in your own voice."
        description="Select the exact application, then create a factual response ready to paste."
      />
      <div className="talk-grid">
        <aside className="panel padded context-panel">
          <div className="section-head">
            <div>
              <h2>Answer context</h2>
              <p>Nothing is assumed silently.</p>
            </div>
          </div>
          <label className="field">
            <span>Company and role</span>
            <select value={key} onChange={(e) => setKey(e.target.value)}>
              <option value="">Choose saved application…</option>
              {contexts.data?.items.map((x) => (
                <option key={x.key} value={x.key}>
                  {x.company} · {x.role}
                </option>
              ))}
            </select>
          </label>
          <div className="context-card">
            {selected ? (
              <>
                <b>{selected.company}</b>
                <span>{selected.role}</span>
                <small>
                  {selected.has_humanized
                    ? "Humanized résumé"
                    : "Optimized résumé"}
                </small>
              </>
            ) : (
              <>
                <b>No context selected</b>
                <span>Choose an application to continue.</span>
              </>
            )}
          </div>
          <label className="field">
            <span>Writing style</span>
            <select value={tone} onChange={(e) => setTone(e.target.value)}>
              <option value="balanced">Natural and balanced</option>
              <option value="concise">Short and direct</option>
              <option value="warm">Warm and conversational</option>
            </select>
          </label>
          <ul className="trust-list">
            <li>IELTS 6.5-level Indian English</li>
            <li>Facts restricted to selected résumé</li>
            <li>No inflated claims or AI clichés</li>
          </ul>
        </aside>
        <section className="panel answer-panel">
          <div className="answer-top">
            <div>
              <h2>Application answer</h2>
              <p>Ask exactly what the form asks.</p>
            </div>
            <span className="secure">Context locked</span>
          </div>
          <div className="chips">
            {presets.map((x) => (
              <button key={x} onClick={() => setQuestion(x)}>
                {x}
              </button>
            ))}
          </div>
          <div className="composer">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Paste the application question here…"
            />
            <div>
              <span>90–150 words · factual · natural</span>
              <button
                className="button primary"
                disabled={!key || !question.trim() || generate.isPending}
                onClick={() => generate.mutate()}
              >
                {generate.isPending ? "Writing…" : "Generate answer"}
              </button>
            </div>
          </div>
          <div className={`answer ${!answer ? "answer-empty" : ""}`}>
            {generate.isError
              ? "Could not create the answer. Please try again."
              : answer || "Your copy-ready answer will appear here."}
            {answer && (
              <button
                className="copy"
                onClick={() => navigator.clipboard.writeText(answer)}
              >
                Copy
              </button>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function MasterMind() {
  const [session, setSession] = useState("");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);
  const [persona, setPersona] = useState("Career strategist");
  const start = async () => {
    const form = new FormData();
    form.set("persona", persona);
    form.set("purpose", "career reasoning");
    const d = await api<{ session: { id: string } }>("/api/mastermind/start", {
      method: "POST",
      body: form,
    });
    setSession(d.session.id);
    setMessages([]);
    return d.session.id;
  };
  const chat = useMutation({
    mutationFn: async () => {
      const id = session || (await start());
      const form = new FormData();
      form.set("session_id", id);
      form.set("prompt", prompt);
      form.set("persona", persona);
      form.set("tone", "balanced");
      form.set("model", "gpt-5.4-nano");
      return api<{ reply: string }>("/api/mastermind/chat", {
        method: "POST",
        body: form,
      });
    },
    onMutate: () => {
      setMessages((v) => [...v, { role: "user", content: prompt }]);
      setPrompt("");
    },
    onSuccess: (d) =>
      setMessages((v) => [...v, { role: "assistant", content: d.reply }]),
  });
  return (
    <div className="page">
      <PageTitle
        eyebrow="Career intelligence"
        title="Think through the next move."
        description="A focused reasoning workspace for job-search strategy, interview preparation, and technical decisions."
        action={
          <button className="button ghost" onClick={() => start()}>
            <Sparkles size={16} />
            New session
          </button>
        }
      />
      <div className="mm-grid">
        <aside className="panel padded">
          <div className="section-head">
            <div>
              <h2>Conversation setup</h2>
              <p>Keep the guidance focused.</p>
            </div>
          </div>
          <label className="field">
            <span>Advisor mode</span>
            <select
              value={persona}
              onChange={(e) => setPersona(e.target.value)}
            >
              <option>Career strategist</option>
              <option>Technical interviewer</option>
              <option>Resume reviewer</option>
              <option>Hiring manager</option>
            </select>
          </label>
          <div className="context-card">
            <b>{session ? "Session active" : "Ready to begin"}</b>
            <span>
              {session || "A session starts with your first message."}
            </span>
          </div>
          <ul className="trust-list">
            <li>Compact, practical reasoning</li>
            <li>Career and technical focus</li>
            <li>Conversation history preserved</li>
          </ul>
        </aside>
        <section className="panel chat-panel">
          <div className="chat-stream">
            {!messages.length ? (
              <Empty
                title="Start a focused conversation"
                copy="Ask about positioning, interviews, trade-offs, or your next application."
              />
            ) : (
              messages.map((m, i) => (
                <div className={`message ${m.role}`} key={i}>
                  <span>{m.role === "user" ? "You" : "ASTRA"}</span>
                  <p>{m.content}</p>
                </div>
              ))
            )}
            {chat.isPending && (
              <div className="message assistant">
                <span>ASTRA</span>
                <p className="thinking">Thinking through this…</p>
              </div>
            )}
            {chat.isError && (
              <div className="inline-error">{chat.error.message}</div>
            )}
          </div>
          <div className="chat-compose">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  (e.metaKey || e.ctrlKey) &&
                  prompt.trim()
                )
                  chat.mutate();
              }}
              placeholder="Ask MasterMind…"
            />
            <button
              className="button primary"
              disabled={!prompt.trim() || chat.isPending}
              onClick={() => chat.mutate()}
            >
              Send
            </button>
            <small>⌘ Enter to send</small>
          </div>
        </section>
      </div>
    </div>
  );
}

function SuperHuman() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [tone, setTone] = useState("balanced");
  const [mode, setMode] = useState("paragraph");
  const rewrite = useMutation({
    mutationFn: () =>
      api<{ rewritten: string }>("/api/superhuman/rewrite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: input,
          mode,
          tone,
          latex_safe: mode === "resume",
          constraints: { no_fabrication: true, keep_metrics: true },
          max_len: 4000,
        }),
      }),
    onSuccess: (d) =>
      setOutput(typeof d.rewritten === "string" ? d.rewritten : input),
  });
  return (
    <div className="page">
      <PageTitle
        eyebrow="Natural writing"
        title="Polish the wording, keep the facts."
        description="Improve clarity and flow without changing metrics, experience, or meaning."
        action={
          <button
            className="button ghost"
            onClick={() => {
              setInput("");
              setOutput("");
            }}
          >
            Clear
          </button>
        }
      />
      <section className="panel humanizer">
        <div className="humanizer-controls">
          <label className="field">
            <span>Content type</span>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="paragraph">Paragraph</option>
              <option value="resume">Resume bullet</option>
              <option value="coverletter">Cover letter</option>
            </select>
          </label>
          <label className="field">
            <span>Tone</span>
            <select value={tone} onChange={(e) => setTone(e.target.value)}>
              <option value="balanced">Balanced</option>
              <option value="formal">Formal</option>
              <option value="conversational">Conversational</option>
              <option value="confident">Confident</option>
            </select>
          </label>
          <div className="humanizer-note">
            <FileCheck2 size={16} />
            <span>Facts and metrics remain protected.</span>
          </div>
        </div>
        <div className="editor-grid">
          <div className="editor-pane">
            <header>
              <div>
                <b>Original</b>
                <span>{input.length} characters</span>
              </div>
            </header>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Paste the text you want to improve…"
            />
          </div>
          <div className="editor-pane">
            <header>
              <div>
                <b>Refined</b>
                <span>{output.length} characters</span>
              </div>
              {output && (
                <button onClick={() => navigator.clipboard.writeText(output)}>
                  Copy
                </button>
              )}
            </header>
            <div className={`editor-output ${!output ? "muted-output" : ""}`}>
              {rewrite.isPending
                ? "Refining the language…"
                : rewrite.isError
                  ? `Could not humanize: ${rewrite.error.message}`
                  : output || "Your refined text will appear here."}
            </div>
          </div>
        </div>
        <div className="humanizer-foot">
          <span>Natural language · no fabrication · original meaning</span>
          <button
            className="button primary"
            disabled={!input.trim() || rewrite.isPending}
            onClick={() => rewrite.mutate()}
          >
            <Zap size={16} />
            {rewrite.isPending ? "Refining…" : "Humanize text"}
          </button>
        </div>
      </section>
    </div>
  );
}

function usePdfUrl(base64?: string) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!base64) {
      setUrl("");
      return;
    }
    let objectUrl = "";
    try {
      const raw = base64.includes("base64,")
        ? base64.split("base64,").pop() || ""
        : base64;
      const binary = atob(raw.replace(/\s/g, ""));
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      objectUrl = URL.createObjectURL(
        new Blob([bytes], { type: "application/pdf" }),
      );
      setUrl(objectUrl);
    } catch {
      setUrl("");
    }
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [base64]);
  return url;
}

function InternetJobs() {
  const [query, setQuery] = useState("");
  const [region, setRegion] = useState("all");
  const [category, setCategory] = useState("all");
  const [companyTier, setCompanyTier] = useState<"all" | "major">("all");
  const [sponsorship, setSponsorship] = useState("all");
  const [freshness, setFreshness] = useState(0);
  const [workplace, setWorkplace] = useState("all");
  const [view, setView] = useState<"eligible" | "filtered" | "all">("eligible");
  const [companiesOpen, setCompaniesOpen] = useState(false);
  const [companyQuery, setCompanyQuery] = useState("");
  const [selected, setSelected] = useState<InternetJob | null>(null);
  const qc = useQueryClient();
  const jobs = useQuery({
    queryKey: ["internet-jobs", query, region, category, companyTier, sponsorship, freshness, workplace, view],
    queryFn: () => api<InternetJobsResponse>(`/api/internet-jobs?q=${encodeURIComponent(query)}&region=${encodeURIComponent(region)}&category=${encodeURIComponent(category)}&company_tier=${companyTier}&sponsorship=${sponsorship}&freshness_days=${freshness}&workplace=${workplace}&view=${view}&hide_applied=${view === "eligible"}&limit=2000`),
    refetchInterval: (result) => result.state.data?.runtime.running ? 1500 : 30000,
  });
  const refresh = useMutation({
    mutationFn: () => api("/api/internet-jobs/refresh", { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["internet-jobs"] });
      notify("Internet job search started", "info");
    },
    onError: (error) => notify(error.message, "error"),
  });
  const discoverySettings = useQuery({
    queryKey: ["internet-job-settings"],
    queryFn: () => api<{ enabled: boolean; interval_hours: number; minimum_score: number; quality: Record<string, number> }>("/api/internet-jobs/settings"),
  });
  const sponsorCompanies = useQuery({
    queryKey: ["internet-sponsor-companies", companyQuery],
    queryFn: () => api<{ items: { company: string; legal_employer: string; tech_lca_positions: number; direct_feed: boolean; live_in_astra: boolean; career_url?: string; ats: string }[]; total: number }>(`/api/internet-jobs/companies?q=${encodeURIComponent(companyQuery)}`),
    enabled: companiesOpen,
  });
  const savedSearches = useQuery({
    queryKey: ["internet-saved-searches"],
    queryFn: () => api<{ items: { id: string; name: string; q: string; region: string; category: string; minimum_score: number }[]; alerts: { id: string; name: string; count: number }[] }>("/api/internet-jobs/saved-searches"),
  });
  const saveCurrentSearch = useMutation({
    mutationFn: () => api("/api/internet-jobs/saved-searches", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      name: query || `${category === "all" ? "All related roles" : category} · ${region === "all" ? "USA" : region}`,
      q: query, region, category, minimum_score: discoverySettings.data?.minimum_score || 60,
    }) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["internet-saved-searches"] }); notify("Search and new-job alert saved"); },
    onError: (error) => notify(error.message, "error"),
  });
  const deleteSavedSearch = useMutation({
    mutationFn: (id: string) => api(`/api/internet-jobs/saved-searches/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["internet-saved-searches"] }),
  });
  const saveDiscoverySettings = useMutation({
    mutationFn: (payload: Record<string, unknown>) => api("/api/internet-jobs/settings", {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["internet-job-settings"] }); notify("Discovery schedule updated"); },
    onError: (error) => notify(error.message, "error"),
  });
  const feedback = useMutation({
    mutationFn: ({ job, action }: { job: InternetJob; action: "restore" | "hide" | "correct" }) => api(`/api/internet-jobs/${job.id}/feedback`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, note: action === "restore" ? "Filter marked incorrect from Internet Jobs" : "Manual Internet Jobs correction" }),
    }),
    onSuccess: (_, value) => {
      qc.invalidateQueries({ queryKey: ["internet-jobs"] });
      setSelected(null);
      notify(value.action === "restore" ? "Job restored and correction recorded" : "Job hidden and feedback recorded");
    },
    onError: (error) => notify(error.message, "error"),
  });
  useEffect(() => {
    if (jobs.data && !jobs.data.updated_at && !jobs.data.runtime.running && !refresh.isPending) refresh.mutate();
  }, [jobs.data?.updated_at, jobs.data?.runtime.running]);
  const useJob = (job: InternetJob) => {
    localStorage.setItem("astra-studio-draft", job.description || "");
    localStorage.setItem("astra-studio-job", JSON.stringify({
      source: job.source, id: job.id, title: job.title, company: job.company,
      url: job.url, application_url: job.application_url,
      requisition_id: job.requisition_id || "",
    }));
    location.hash = "#/studio";
    notify("Internet job sent to Resume Studio");
  };
  const data = jobs.data;
  const items = data?.items || [];
  const categories = ["all", "Machine Learning & AI", "LLM & Inference", "Computer Vision", "Software Engineering", "Data"];
  const salary = (value: unknown) => {
    if (!value) return "";
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map((item) => typeof item === "object" && item ? String((item as Record<string, unknown>).title || "") : "").filter(Boolean).join(" · ");
    if (typeof value === "object") {
      const item = value as Record<string, unknown>;
      return item.min && item.max ? `$${Number(item.min).toLocaleString()}–$${Number(item.max).toLocaleString()}` : "";
    }
    return "";
  };
  return <div className="page">
    <PageTitle eyebrow="Internet job discovery" title="Fresh AI jobs beyond Jobright." description="Search direct public employer feeds across the United States, with California and Bay Area roles ranked first. Applied jobs and degree-ineligible roles are removed." action={<button className="button primary" disabled={refresh.isPending || data?.runtime.running} onClick={() => refresh.mutate()}><RefreshCw size={16}/>{data?.runtime.running ? "Searching…" : data?.updated_at ? "Refresh internet jobs" : "Search internet jobs"}</button>} />
    <div className="jobright-summary panel">
      <div><span>Eligible unapplied jobs</span><strong>{(data?.eligible_total || data?.total || 0).toLocaleString()}</strong></div>
      <div><span>Bay Area</span><strong>{(data?.counts["Bay Area"] || 0).toLocaleString()}</strong></div>
      <div><span>New since last scan</span><strong>{(data?.new_jobs || 0).toLocaleString()}</strong></div>
      <div className={data?.runtime.error ? "sync-error" : "sync-state"}><span>Status</span><strong>{data?.runtime.message || "Ready to search"}</strong></div>
    </div>
    {data?.sponsor_registry ? <div className="sponsor-coverage panel"><div><b>{data.sponsor_registry.targets} sponsor-backed US tech targets</b><small>{data.sponsor_registry.connected_feeds} live public career feeds connected · {data.sponsor_registry.direct_feeds} verified direct-feed employers in the registry</small></div><span>Evidence: {data.sponsor_registry.evidence || "DOL certified LCA history"}</span><button onClick={() => setCompaniesOpen((value) => !value)}>{companiesOpen ? "Close directory" : "View all 300 companies"}</button><p>{data.sponsor_registry.disclaimer}</p></div> : null}
    {companiesOpen ? <section className="sponsor-directory panel"><header><div><b>Top 300 technology sponsor directory</b><small>Google, Microsoft, Apple and other custom career sites remain visible even when they do not provide a standard ATS feed.</small></div><label className="search"><Search size={15}/><input value={companyQuery} onChange={(event) => setCompanyQuery(event.target.value)} placeholder="Search all 300 companies"/></label></header><div className="sponsor-company-grid">{sponsorCompanies.isLoading ? <Skeleton rows={6}/> : (sponsorCompanies.data?.items || []).map((company) => <article key={`${company.company}-${company.legal_employer}`}><div><b>{company.company}</b><small>{company.tech_lca_positions.toLocaleString()} certified technical LCA positions</small></div><span className={company.live_in_astra ? "live" : "watch"}>{company.live_in_astra ? "Live in ASTRA" : "Career-site watchlist"}</span>{company.career_url ? <a href={company.career_url} target="_blank" rel="noreferrer">Open careers <ExternalLink size={12}/></a> : null}</article>)}</div></section> : null}
    {data?.active_filters ? <div className="internet-profile-bar"><b>Eligibility profile active</b><span>0–{data.active_filters.years_experience} years experience</span><span>{data.active_filters.degree}</span><span>{data.active_filters.currently_student === "yes" ? "Current student roles allowed" : "No currently-pursuing roles"}</span><span>{data.active_filters.security_clearance === "yes" ? "Clearance roles allowed" : "No security-clearance roles"}</span><span>{data.active_filters.requires_sponsorship === "yes" ? "Must support future sponsorship" : "No sponsorship filter"}</span><a href="#/operations">Edit profile</a></div> : null}
    {discoverySettings.data ? <div className="internet-schedule panel"><div><b>Automatic discovery</b><small>{discoverySettings.data.enabled ? `Runs every ${discoverySettings.data.interval_hours} hours and reports only genuinely new jobs.` : "Automatic discovery is paused."}</small></div><label><input type="checkbox" checked={discoverySettings.data.enabled} onChange={(event) => saveDiscoverySettings.mutate({ enabled: event.target.checked })}/> Enabled</label><select value={discoverySettings.data.interval_hours} onChange={(event) => saveDiscoverySettings.mutate({ interval_hours: Number(event.target.value) })}><option value={3}>Every 3 hours</option><option value={6}>Every 6 hours</option><option value={12}>Every 12 hours</option><option value={24}>Daily</option></select></div> : null}
    {data?.runtime.running ? <div className="import-banner"><span className="progress-ring"/><div><b>Searching employer career systems in parallel</b><p>ASTRA is downloading public descriptions, filtering education restrictions and removing jobs already represented in your application library.</p></div></div> : null}
    <div className="internet-filters panel">
      <label className="search"><Search size={17}/><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search title, company, skill or job description"/></label>
      <select value={region} onChange={(event) => { setRegion(event.target.value); setSelected(null); }}><option value="all">All USA · Bay Area first</option><option value="Bay Area">Bay Area only</option><option value="California">California only</option><option value="United States">Other USA</option></select>
      <select value={category} onChange={(event) => { setCategory(event.target.value); setSelected(null); }}>{categories.map((item) => <option value={item} key={item}>{item === "all" ? "All related roles" : item}</option>)}</select>
      <button className={`button ghost ${companyTier === "major" ? "active" : ""}`} onClick={() => { setCompanyTier((value) => value === "major" ? "all" : "major"); setSelected(null); }}>{companyTier === "major" ? "Big Tech only ✓" : "Big Tech only"}</button>
      <select value={sponsorship} onChange={(event) => setSponsorship(event.target.value)}><option value="all">Any sponsorship evidence</option><option value="confirmed">Role confirms sponsorship</option><option value="history">Confirmed or company history</option><option value="unknown">Sponsorship unknown</option></select>
      <select value={freshness} onChange={(event) => setFreshness(Number(event.target.value))}><option value={0}>Any posting date</option><option value={3}>Last 3 days</option><option value={7}>Last 7 days</option><option value={14}>Last 14 days</option><option value={30}>Last 30 days</option></select>
      <select value={workplace} onChange={(event) => setWorkplace(event.target.value)}><option value="all">Any workplace</option><option value="remote">Remote</option><option value="hybrid">Hybrid</option><option value="onsite">Onsite</option></select>
      <button className="button ghost" onClick={() => saveCurrentSearch.mutate()}>Save search</button>
    </div>
    {(savedSearches.data?.items.length || savedSearches.data?.alerts.length) ? <div className="saved-search-row"><b>Saved alerts</b>{savedSearches.data?.alerts.map((alert) => <span className="new-alert" key={alert.id}>{alert.name} · {alert.count} new</span>)}{savedSearches.data?.items.map((saved) => <span key={saved.id}><button onClick={() => { setQuery(saved.q); setRegion(saved.region); setCategory(saved.category); }}>{saved.name}</button><button aria-label={`Delete ${saved.name}`} onClick={() => deleteSavedSearch.mutate(saved.id)}>×</button></span>)}</div> : null}
    <div className="internet-view-tabs">
      <button className={view === "eligible" ? "active" : ""} onClick={() => { setView("eligible"); setSelected(null); }}>Recommended <span>{data?.eligible_total || 0}</span></button>
      <button className={view === "filtered" ? "active" : ""} onClick={() => { setView("filtered"); setSelected(null); }}>Filtered with reasons <span>{data?.filtered_total || 0}</span></button>
      <button className={view === "all" ? "active" : ""} onClick={() => { setView("all"); setSelected(null); }}>All discovered</button>
    </div>
    <div className="jobright-grid internet-grid">
      <section className="panel job-feed"><div className="toolbar"><span className="feed-count">{items.length.toLocaleString()} jobs shown · newest source updates first</span></div><div className="job-list">
        {jobs.isLoading ? <Skeleton rows={9}/> : jobs.isError ? <Empty title="Internet search is unavailable" copy="Restart ASTRA and try the search again."/> : items.length ? items.map((job) => <button key={job.id} className={selected?.id === job.id ? "selected" : ""} onClick={() => setSelected(job)}><span className="company-mark">{job.company.slice(0, 2).toUpperCase()}</span><span><b>{job.title}</b><small>{job.company} · {job.location || "USA"}</small><time>{job.region} · {(job.source_names || [job.source]).join(" + ")}{job.updated_at ? ` · Updated ${fmt(job.updated_at)}` : ""}</time></span><em className={job.filter_decision?.eligible ? "" : "filtered-tag"}>{job.filter_decision?.eligible ? `${job.rank_score || job.filter_decision.score}% fit` : "Filtered"}</em></button>) : <Empty title={data?.updated_at ? "No jobs in this view" : "Search the public job internet"} copy={view === "filtered" ? "No relevant postings were rejected by your current eligibility profile." : data?.updated_at ? "Try a broader role or region. Previously applied jobs remain hidden." : "Start a search to import relevant roles from public employer feeds."}/>}
      </div></section>
      <aside className="panel job-detail">{selected ? <><div className="job-detail-head"><span className="eyebrow">{selected.region} · {(selected.source_names || [selected.source]).join(" + ")}</span><h2>{selected.title}</h2><p>{selected.company} · {selected.location || "United States"}</p><div className="detail-tags"><span>{selected.category}</span><span>{selected.rank_score || selected.filter_decision?.score || 0}% fit</span><span>Link {selected.link_status || "unchecked"}</span>{selected.sponsor_history ? <span>{selected.tech_lca_positions || 0} recent technical LCA positions</span> : null}{selected.duplicate_count && selected.duplicate_count > 1 ? <span>{selected.duplicate_count} sources merged</span> : null}{selected.workplace_type ? <span>{selected.workplace_type}</span> : null}{selected.requisition_id ? <span>Req {selected.requisition_id}</span> : null}{salary(selected.salary) ? <span>{salary(selected.salary)}</span> : null}</div></div><div className="job-detail-body">
        <section className={`filter-explanation ${selected.filter_decision?.eligible ? "eligible" : "rejected"}`}><h3>{selected.filter_decision?.eligible ? "Why ASTRA recommends this" : "Why ASTRA filtered this"}</h3>{selected.filter_decision?.reasons?.length ? selected.filter_decision.reasons.map((reason) => <div key={reason.code}><b>{reason.message}</b><blockquote>{reason.evidence}</blockquote></div>) : <p>No hard eligibility conflict was found. Ranking combines experience, technical coverage, role relevance, location and freshness.</p>}{selected.filter_decision?.technical_fit !== undefined ? <small>Technical requirement coverage: {selected.filter_decision.technical_fit}%{selected.filter_decision.missing_skills?.length ? ` · Check: ${selected.filter_decision.missing_skills.join(", ")}` : ""}</small> : null}{selected.filter_decision?.overridden ? <small>Manual correction active</small> : null}</section>
        <section className="internet-score-card"><div><h3>Application value</h3><strong>{selected.rank_score || 0}%</strong></div><div className="score-components">{Object.entries(selected.rank_components || {}).map(([name, score]) => <span key={name}><small>{name.replace("_", " ")}</small><b>{score}%</b></span>)}</div><p><b>Sponsorship: {selected.sponsorship?.status.replaceAll("_", " ") || "unknown"}</b> · {selected.sponsorship?.reason || "No evidence available."}</p>{selected.application_policy?.cooldown_days ? <p>Company application cooldown detected: {selected.application_policy.cooldown_days} days.</p> : null}{selected.data_quality && (!selected.data_quality.valid_url || !selected.data_quality.complete_description) ? <p className="quality-warning">Document warning: {!selected.data_quality.valid_url ? "invalid application URL" : "partial description"}.</p> : null}</section>
        <section className="job-section"><h3>Complete job description</h3><div className="description-copy">{(selected.description || "Description is not available.").split("\n").filter(Boolean).map((line, index) => <p key={index}>{line}</p>)}</div></section></div><div className="job-detail-actions"><a className="button ghost" href={selected.url} target="_blank" rel="noreferrer"><ExternalLink size={15}/>Employer posting</a>{selected.filter_decision?.eligible ? <><button className="button ghost" onClick={() => feedback.mutate({ job: selected, action: "hide" })}>Hide job</button><button className="button primary" onClick={() => useJob(selected)}><Command size={15}/>Generate tailored résumé</button></> : <button className="button primary" onClick={() => feedback.mutate({ job: selected, action: "restore" })}>Filter is wrong · restore</button>}</div></> : <Empty title="Choose an internet job" copy="Review its fit score, exact eligibility evidence and complete employer description."/>}</aside>
    </div>
    {data?.sources?.length ? <details className="source-coverage panel"><summary>Source health · {data.sources.filter((item) => item.ok).length} of {data.sources.length} feeds healthy</summary><div>{data.sources.map((item) => <span className={item.ok ? "source-ok" : "source-error"} key={`${item.source}-${item.company}`}><b>{item.company}</b><small>{item.ok ? `${item.jobs} matching · ${item.source}${item.cached ? " · cached" : item.duration_ms ? ` · ${item.duration_ms}ms` : ""}` : `Failure ${item.consecutive_failures || 1} · ${item.error || item.source}`}</small>{item.last_success ? <small>Last success {fmt(item.last_success)}</small> : null}</span>)}</div></details> : null}
  </div>;
}

function JobrightJobs() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState<JobrightJob | null>(null);
  const qc = useQueryClient();
  const [agentOpen, setAgentOpen] = useState(false);
  const [agentForm, setAgentForm] = useState<AgentResponse["settings"] | null>(
    null,
  );
  const jobs = useQuery({
    queryKey: ["jobright-jobs", query, category],
    queryFn: () =>
      api<JobrightResponse>(
        `/api/jobright?q=${encodeURIComponent(query)}&category=${encodeURIComponent(category)}&relevant_only=true`,
      ),
    refetchInterval: (q) => (q.state.data?.state.running ? 1500 : false),
  });
  const connect = useMutation({
    mutationFn: () =>
      api<{ message: string }>("/api/jobright/connect", { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobright-jobs"] });
      notify("Jobright sign-in window opened", "info");
    },
    onError: (e) => notify(e.message, "error"),
  });
  const details = useMutation({
    mutationFn: (job: JobrightJob) =>
      api<JobrightJob>(`/api/jobright/${job.id}/details`, { method: "POST" }),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ["jobright-jobs"] });
      if (job.education_ineligible) {
        setSelected(null);
        notify(
          "Hidden because this role requires current student or PhD eligibility",
          "info",
        );
      } else setSelected(job);
    },
    onError: (e) => notify(e.message, "error"),
  });
  const selectJob = (job: JobrightJob) => {
    setSelected(job);
    api(`/api/jobright/${job.id}/view`, { method: "POST" })
      .then(() => qc.invalidateQueries({ queryKey: ["jobright-jobs"] }))
      .catch(() => {});
  };
  useEffect(() => {
    if (
      selected?.id &&
      (selected.description || "").length < 300 &&
      !details.isPending
    )
      details.mutate(selected);
  }, [selected?.id]);
  const useJob = async (job: JobrightJob) => {
    try {
      const ready =
        (job.description || "").length >= 300
          ? job
          : await api<JobrightJob>(`/api/jobright/${job.id}/details`, {
              method: "POST",
            });
      localStorage.setItem("astra-studio-draft", ready.description || "");
      localStorage.setItem(
        "astra-studio-job",
        JSON.stringify({
          source: "Jobright",
          id: ready.id,
          title: ready.title,
          company: ready.company,
          url: ready.url,
          application_url: ready.application_url,
        }),
      );
      location.hash = "#/studio";
      notify("Full job description sent to Resume Studio");
    } catch (e) {
      notify(
        e instanceof Error ? e.message : "Could not load job details",
        "error",
      );
    }
  };
  const openJob = useMutation({
    mutationFn: ({
      job,
      target,
    }: {
      job: JobrightJob;
      target: "jobright" | "application";
    }) =>
      api<{ url: string }>(`/api/jobright/${job.id}/open?target=${target}`, {
        method: "POST",
      }),
    onSuccess: (_, v) =>
      notify(
        v.target === "application"
          ? "Application opened in your browser"
          : "Jobright page opened",
        "info",
      ),
    onError: (e) => notify(e.message, "error"),
  });
  const agent = useQuery({
    queryKey: ["jobright-agent"],
    queryFn: () => api<AgentResponse>("/api/jobright/agent"),
    refetchInterval: (q) => (q.state.data?.runtime.running ? 1500 : 15000),
  });
  useEffect(() => {
    if (agent.data && !agentForm) setAgentForm(agent.data.settings);
  }, [agent.data, agentForm]);
  const saveAgent = useMutation({
    mutationFn: () =>
      api<AgentResponse>("/api/jobright/agent", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(agentForm),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobright-agent"] });
      notify("Application agent settings saved");
    },
    onError: (e) => notify(e.message, "error"),
  });
  const runAgent = useMutation({
    mutationFn: () => api("/api/jobright/agent/run", { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobright-agent"] });
      notify("Daily application run started", "info");
    },
    onError: (e) => notify(e.message, "error"),
  });
  const setProfile = (key: keyof AgentProfile, value: string) =>
    setAgentForm((v) =>
      v ? { ...v, profile: { ...v.profile, [key]: value } } : v,
    );
  const state = jobs.data?.state;
  const items = jobs.data?.items || [];
  const categories = [
    "all",
    "Machine Learning & AI",
    "LLM & Inference",
    "Computer Vision",
    "Software Engineering",
    "Data",
  ];
  const facts = (job: JobrightJob) =>
    (job.card_text || "")
      .split("\n")
      .map((x) => x.trim())
      .filter(
        (x) =>
          x &&
          ![
            "company-logo",
            "banner",
            "/",
            "Why this job is a match",
            job.title,
            job.company,
            job.summary,
          ].includes(x) &&
          !/^\d+%$/.test(x),
      )
      .slice(0, 16);
  return (
    <div className="page">
      <PageTitle
        eyebrow="Job discovery"
        title="Your Jobright feed, inside ASTRA."
        description="Connect your Jobright web account, import every job shown in your feed, and send the complete description directly to Resume Studio."
        action={
          <button
            className="button primary"
            disabled={connect.isPending || state?.running}
            onClick={() => connect.mutate()}
          >
            <RefreshCw size={16} />
            {state?.running
              ? "Importing…"
              : jobs.data?.total
                ? "Refresh Jobright"
                : "Connect Jobright"}
          </button>
        }
      />
      <div className="jobright-summary panel">
        <div>
          <span>Connected account</span>
          <strong>{jobs.data?.account || "kadali18@umd.edu"}</strong>
        </div>
        <div>
          <span>Relevant roles</span>
          <strong>
            {(jobs.data?.relevant_total || 0).toLocaleString()} of{" "}
            {(jobs.data?.all_total || 0).toLocaleString()}
          </strong>
        </div>
        <div>
          <span>Last refresh</span>
          <strong>
            {jobs.data?.updated_at
              ? fmt(jobs.data.updated_at)
              : "Not connected"}
          </strong>
        </div>
        <div className={state?.phase === "error" ? "sync-error" : "sync-state"}>
          <span>Status</span>
          <strong>{state?.message || "Ready"}</strong>
        </div>
      </div>
      {state?.running ? (
        <div className="import-banner">
          <span className="progress-ring" />
          <div>
            <b>{state.message}</b>
            <p>
              Use the private Chrome window opened by ASTRA. Sign in normally;
              your password is never stored by ASTRA.
            </p>
          </div>
        </div>
      ) : null}
      <section className="panel agent-control">
        <button
          className="agent-control-head"
          onClick={() => setAgentOpen((v) => !v)}
        >
          <span className={agent.data?.settings.enabled ? "agent-live" : ""}>
            <Zap size={16} />
          </span>
          <div>
            <b>Daily application agent</b>
            <small>
              {agent.data?.runtime.running
                ? `${agent.data.runtime.message} · ${agent.data.runtime.current_job || ""}`
                : agent.data?.settings.enabled
                  ? `Runs daily at ${agent.data.settings.run_at} · ${agent.data.settings.daily_limit ? `up to ${agent.data.settings.daily_limit} applications` : "all remaining relevant jobs"}`
                  : "Complete your profile and enable the schedule."}
            </small>
          </div>
          <em>
            {agent.data?.runtime.running
              ? "Running"
              : agent.data?.settings.enabled
                ? "Enabled"
                : "Setup required"}
          </em>
          <ChevronRight size={16} />
        </button>
        {agentOpen && agentForm && (
          <div className="agent-settings">
            <div className="agent-fields">
              <label className="field">
                <span>Phone</span>
                <input
                  value={agentForm.profile.phone}
                  onChange={(e) => setProfile("phone", e.target.value)}
                  placeholder="+1…"
                />
              </label>
              <label className="field">
                <span>Location</span>
                <input
                  value={agentForm.profile.location}
                  onChange={(e) => setProfile("location", e.target.value)}
                  placeholder="City, State"
                />
              </label>
              <label className="field">
                <span>LinkedIn</span>
                <input
                  value={agentForm.profile.linkedin}
                  onChange={(e) => setProfile("linkedin", e.target.value)}
                  placeholder="https://linkedin.com/in/…"
                />
              </label>
              <label className="field">
                <span>Portfolio / GitHub</span>
                <input
                  value={agentForm.profile.portfolio}
                  onChange={(e) => setProfile("portfolio", e.target.value)}
                  placeholder="https://…"
                />
              </label>
              <label className="field">
                <span>Authorized to work in the US?</span>
                <select
                  value={agentForm.profile.authorized_to_work}
                  onChange={(e) =>
                    setProfile("authorized_to_work", e.target.value)
                  }
                >
                  <option value="">Choose…</option>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </label>
              <label className="field">
                <span>Require current/future sponsorship?</span>
                <select
                  value={agentForm.profile.requires_sponsorship}
                  onChange={(e) =>
                    setProfile("requires_sponsorship", e.target.value)
                  }
                >
                  <option value="">Choose…</option>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </label>
              <label className="field">
                <span>Daily run time</span>
                <input
                  type="time"
                  value={agentForm.run_at}
                  onChange={(e) =>
                    setAgentForm((v) =>
                      v ? { ...v, run_at: e.target.value } : v,
                    )
                  }
                />
              </label>
              <label className="field">
                <span>Daily application limit</span>
                <input
                  type="number"
                  min="0"
                  max="10000"
                  value={agentForm.daily_limit}
                  onChange={(e) =>
                    setAgentForm((v) =>
                      v ? { ...v, daily_limit: Number(e.target.value) } : v,
                    )
                  }
                />
                <small>0 means every remaining relevant job.</small>
              </label>
            </div>
            <div className="agent-options">
              <label className="switch-row">
                <span>
                  <b>Enable daily agent</b>
                  <small>Runs while ASTRA is open in the background.</small>
                </span>
                <input
                  type="checkbox"
                  checked={agentForm.enabled}
                  onChange={(e) =>
                    setAgentForm((v) =>
                      v ? { ...v, enabled: e.target.checked } : v,
                    )
                  }
                />
              </label>
              <label className="switch-row">
                <span>
                  <b>Submit automatically</b>
                  <small>
                    Unknown required questions and CAPTCHAs are paused, never
                    guessed.
                  </small>
                </span>
                <input
                  type="checkbox"
                  checked={agentForm.auto_submit}
                  onChange={(e) =>
                    setAgentForm((v) =>
                      v ? { ...v, auto_submit: e.target.checked } : v,
                    )
                  }
                />
              </label>
            </div>
            <div className="agent-actions">
              <button
                className="button ghost"
                onClick={() => runAgent.mutate()}
                disabled={agent.data?.runtime.running || runAgent.isPending}
              >
                <Zap size={15} />
                Run all relevant now
              </button>
              <button
                className="button primary"
                onClick={() => saveAgent.mutate()}
                disabled={saveAgent.isPending}
              >
                {saveAgent.isPending ? "Saving…" : "Save agent settings"}
              </button>
            </div>
            {agent.data?.events?.length ? (
              <div className="agent-log">
                <b>Recent activity</b>
                {agent.data.events.slice(0, 5).map((event, i) => (
                  <div key={i}>
                    <span
                      className={`agent-status agent-status-${event.status}`}
                    >
                      {event.status.replaceAll("_", " ")}
                    </span>
                    <p>
                      <strong>
                        {event.company} · {event.title}
                      </strong>
                      <small>{event.message}</small>
                    </p>
                    <time>{fmt(event.at)}</time>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        )}
      </section>
      <div className="role-filters">
        {categories.map((x) => (
          <button
            key={x}
            className={category === x ? "active" : ""}
            onClick={() => {
              setCategory(x);
              setSelected(null);
            }}
          >
            {x === "all" ? "All relevant" : x}
            <span>
              {x === "all"
                ? jobs.data?.relevant_total || 0
                : jobs.data?.category_counts?.[x] || 0}
            </span>
          </button>
        ))}
      </div>
      <div className="jobright-grid">
        <section className="panel job-feed">
          <div className="toolbar">
            <label className="search">
              <Search size={17} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search relevant title, company, or location"
              />
            </label>
            <span className="feed-count">
              {items.length.toLocaleString()} unapplied roles
            </span>
          </div>
          <div className="job-list">
            {jobs.isLoading ? (
              <Skeleton rows={8} />
            ) : jobs.isError ? (
              <Empty
                title="Jobright is unavailable"
                copy="Restart ASTRA and try connecting again."
              />
            ) : items.length ? (
              items.map((job) => (
                <button
                  key={job.id}
                  className={selected?.id === job.id ? "selected" : ""}
                  onClick={() => selectJob(job)}
                >
                  <span className="company-mark">
                    {(job.company || "JR").slice(0, 2).toUpperCase()}
                  </span>
                  <span>
                    <b>{job.title || "Untitled role"}</b>
                    <small>
                      {job.company || "Company"}
                      {job.location ? ` · ${job.location}` : ""}
                    </small>
                    <time>
                      {job.category} ·{" "}
                      {job.last_opened_at
                        ? `Opened ${fmt(job.last_opened_at)}`
                        : job.posted || fmt(job.captured_at)}
                    </time>
                  </span>
                  {job.match ? (
                    <em>{job.match}</em>
                  ) : (
                    <ChevronRight size={15} />
                  )}
                </button>
              ))
            ) : (
              <Empty
                title={
                  jobs.data?.all_total
                    ? "No unapplied roles in this view"
                    : "Connect your Jobright account"
                }
                copy={
                  jobs.data?.all_total
                    ? "Jobs already represented in your PDF résumé library are hidden. Try another category or refresh Jobright for new recommendations."
                    : "ASTRA will import your filtered recommendations after you sign in to Jobright."
                }
                action={
                  !jobs.data?.all_total ? (
                    <button
                      className="button primary"
                      onClick={() => connect.mutate()}
                    >
                      Connect Jobright
                    </button>
                  ) : undefined
                }
              />
            )}
          </div>
        </section>
        <aside className="panel job-detail">
          {selected ? (
            <>
              <div className="job-detail-head">
                <span className="eyebrow">
                  {selected.category || "Relevant role"} · Jobright match
                </span>
                <h2>{selected.title}</h2>
                <p>
                  {selected.company}
                  {selected.location ? ` · ${selected.location}` : ""}
                </p>
                <div className="application-steps">
                  <span className="done">
                    <b>1</b>Job selected
                  </span>
                  <span
                    className={
                      (selected.description || "").length >= 300 ? "done" : ""
                    }
                  >
                    <b>2</b>Description
                  </span>
                  <span>
                    <b>3</b>Résumé
                  </span>
                  <span>
                    <b>4</b>Apply
                  </span>
                </div>
              </div>
              <div className="job-detail-body">
                <div className="detail-tags">
                  {selected.posted && <span>{selected.posted}</span>}
                  {selected.match && <span>{selected.match}</span>}
                  <span>
                    {selected.description_status === "complete"
                      ? "Full description ready"
                      : "Loading complete description…"}
                  </span>
                </div>
                {details.isPending && (
                  <div className="detail-loading">
                    <span className="progress-ring" />
                    <p>Loading the complete Jobright description…</p>
                  </div>
                )}
                {selected.summary && (
                  <section className="job-section">
                    <h3>Overview</h3>
                    <p>{selected.summary}</p>
                  </section>
                )}
                <section className="job-section">
                  <h3>Job information</h3>
                  <div className="fact-grid">
                    {facts(selected).map((fact, i) => (
                      <span key={i}>{fact}</span>
                    ))}
                  </div>
                </section>
                {selected.description_sections &&
                  Object.entries(selected.description_sections)
                    .filter(([, copy]) => copy)
                    .map(([name, copy]) => (
                      <section className="job-section" key={name}>
                        <h3>
                          {name === "required"
                            ? "Required qualifications"
                            : name === "preferred"
                              ? "Preferred qualifications"
                              : name}
                        </h3>
                        <div className="description-copy">
                          {copy
                            .split("\n")
                            .filter(Boolean)
                            .map((line, i) => (
                              <p key={i}>{line.replace(/^\*+|\*+$/g, "")}</p>
                            ))}
                        </div>
                      </section>
                    ))}
              </div>
              <div className="job-detail-actions">
                <button
                  className="button ghost"
                  onClick={() =>
                    openJob.mutate({ job: selected, target: "jobright" })
                  }
                >
                  <ExternalLink size={15} />
                  View in Jobright
                </button>
                <button
                  className="button ghost"
                  disabled={!selected.application_url || openJob.isPending}
                  onClick={() =>
                    openJob.mutate({ job: selected, target: "application" })
                  }
                >
                  <BriefcaseBusiness size={15} />
                  Open application
                </button>
                <button
                  className="button primary"
                  disabled={details.isPending}
                  onClick={() => useJob(selected)}
                >
                  <Command size={15} />
                  Generate tailored résumé
                </button>
              </div>
            </>
          ) : (
            <Empty
              title="Choose a relevant role"
              copy="Review an ML, AI, vision, inference, data, or software-engineering role, then send it directly to Resume Studio."
            />
          )}
        </aside>
      </div>
    </div>
  );
}

function ResumeStudio() {
  const [jd, setJd] = useState(
    () => localStorage.getItem("astra-studio-draft") || "",
  );
  const [humanize, setHumanize] = useState(true);
  const [baseFile, setBaseFile] = useState<File | null>(null);
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const pdfUrl = usePdfUrl(result?.pdf_base64_humanized || result?.pdf_base64);
  useEffect(() => {
    const timer = setTimeout(
      () => localStorage.setItem("astra-studio-draft", jd),
      350,
    );
    return () => clearTimeout(timer);
  }, [jd]);
  const optimize = useMutation({
    mutationFn: async () => {
      const form = new FormData();
      form.set("jd_text", jd);
      form.set("job_description", jd);
      form.set("use_humanize", String(humanize));
      form.set("latex_safe", "true");
      if (baseFile) form.set("base_resume_tex", baseFile);
      return api<OptimizeResult>("/api/optimize/run", {
        method: "POST",
        body: form,
      });
    },
    onSuccess: async (d) => {
      const jobMeta = JSON.parse(localStorage.getItem("astra-studio-job") || "{}");
      setResult(d);
      localStorage.removeItem("astra-studio-draft");
      const history = JSON.parse(
        localStorage.getItem("astra-jd-history") || "[]",
      );
      localStorage.setItem(
        "astra-jd-history",
        JSON.stringify(
          [
            {
              company: d.company || d.company_name,
              role: d.role,
              jd,
              date: new Date().toISOString(),
            },
            ...history,
          ].slice(0, 8),
        ),
      );
      const save = new FormData();
      save.set("company", jobMeta.company || d.company || d.company_name || "Company");
      save.set("role", jobMeta.title || d.role || "Role");
      save.set("jd_text", jd);
      save.set("resume_tex", d.tex_string || "");
      save.set("humanized_tex", d.humanized_tex || d.tex_string || "");
      save.set("pdf_base64", d.pdf_base64 || "");
      save.set("pdf_base64_humanized", d.pdf_base64_humanized || "");
      save.set(
        "fit_score",
        String(d.rating_score ?? Math.round((d.coverage_ratio || 0) * 100)),
      );
      save.set("source", jobMeta.source || "Manual");
      save.set("job_url", jobMeta.url || "");
      save.set("application_url", jobMeta.application_url || "");
      save.set("requisition_id", jobMeta.requisition_id || "");
      await api("/api/context/save", { method: "POST", body: save });
      localStorage.removeItem("astra-studio-job");
      notify("Résumé created and saved");
    },
    onError: (e) => notify(e.message, "error"),
  });
  const score = result
    ? (result.rating_score ?? Math.round((result.coverage_ratio || 0) * 100))
    : null;
  return (
    <div className="page">
      <PageTitle
        eyebrow="Resume Studio"
        title="Tailor your résumé for the role."
        description="Paste the complete job description. ASTRA keeps your existing optimisation pipeline and creates a focused, ATS-ready version."
      />
      <div className="studio-grid">
        <section className="panel studio-form">
          <div className="section-head">
            <div>
              <h2>Job description</h2>
              <p>Include responsibilities, requirements, company, and title.</p>
            </div>
            <span className="secure">Draft saved automatically</span>
          </div>
          <textarea
            value={jd}
            onChange={(e) => setJd(e.target.value)}
            placeholder="Paste the complete job description here…"
          />
          {jd.length > 0 && jd.length < 300 ? (
            <div className="quality-warning">
              This job description looks short. Include the complete
              responsibilities and requirements for a stronger match.
            </div>
          ) : null}
          <div className="studio-options">
            <label className="upload-card">
              <Upload size={18} />
              <span>
                <b>{baseFile?.name || "Use a different base résumé"}</b>
                <small>
                  {baseFile
                    ? "Custom .tex selected"
                    : "Optional · defaults to your saved base résumé"}
                </small>
              </span>
              <input
                type="file"
                accept=".tex,text/plain"
                onChange={(e) => setBaseFile(e.target.files?.[0] || null)}
              />
            </label>
            <label className="switch-row">
              <span>
                <b>Natural-language refinement</b>
                <small>
                  Keep facts and metrics while smoothing the final wording.
                </small>
              </span>
              <input
                type="checkbox"
                checked={humanize}
                onChange={(e) => setHumanize(e.target.checked)}
              />
            </label>
          </div>
          {optimize.isError && (
            <div className="inline-error">
              {optimize.error.message}
              <button onClick={() => optimize.mutate()}>Retry</button>
            </div>
          )}
          <div className="studio-submit">
            <span>
              {jd.length.toLocaleString()} characters · usually 2–5 minutes
            </span>
            <button
              className="button primary"
              disabled={jd.trim().length < 80 || optimize.isPending}
              onClick={() => optimize.mutate()}
            >
              <Sparkles size={16} />
              {optimize.isPending ? "Building résumé…" : "Optimize résumé"}
            </button>
          </div>
        </section>
        <aside className="panel studio-status">
          <div className="section-head">
            <div>
              <h2>{result ? "Résumé ready" : "What happens next"}</h2>
              <p>
                {result
                  ? `${result.company || result.company_name} · ${result.role}`
                  : "A predictable, reviewable workflow."}
              </p>
            </div>
          </div>
          {optimize.isPending ? (
            <div className="progress-state">
              <span className="progress-ring" />
              <b>Matching your experience to the role</b>
              <p>
                ASTRA is analysing the JD, selecting evidence, compiling the
                PDF, and checking the final page count.
              </p>
            </div>
          ) : result ? (
            <>
              <div className="result-metrics">
                <div>
                  <span>Fit score</span>
                  <strong>{score}%</strong>
                </div>
                <div>
                  <span>Final pages</span>
                  <strong>{result.trim_summary?.final_pages || "—"}</strong>
                </div>
                <div>
                  <span>Items trimmed</span>
                  <strong>{result.trim_summary?.items_removed || 0}</strong>
                </div>
              </div>
              <div className="result-actions">
                <a className="button primary" href="#/preview">
                  Review document
                </a>
                {pdfUrl && (
                  <a
                    className="button ghost"
                    href={pdfUrl}
                    download={`${result.company || result.company_name}-${result.role}-resume.pdf`}
                  >
                    Download PDF
                  </a>
                )}
              </div>
            </>
          ) : (
            <ol className="workflow-list">
              <li>
                <b>1</b>
                <span>
                  <strong>Understand the role</strong>
                  <small>Extract requirements and priorities.</small>
                </span>
              </li>
              <li>
                <b>2</b>
                <span>
                  <strong>Match defensible evidence</strong>
                  <small>Use only experience from your base résumé.</small>
                </span>
              </li>
              <li>
                <b>3</b>
                <span>
                  <strong>Compile and validate</strong>
                  <small>
                    Produce the same reliable PDF and source output.
                  </small>
                </span>
              </li>
            </ol>
          )}
        </aside>
      </div>
    </div>
  );
}

function CoverLetters() {
  const contexts = useQuery({
    queryKey: ["contexts", "cover"],
    queryFn: () => api<{ items: Context[] }>("/api/context/list?limit=5000"),
  });
  const [key, setKey] = useState("");
  const [tone, setTone] = useState("balanced");
  const [length, setLength] = useState("standard");
  const [humanize, setHumanize] = useState(true);
  const [result, setResult] = useState<CoverResult | null>(null);
  const [body, setBody] = useState("");
  const [jd, setJd] = useState("");
  const [view, setView] = useState<"pdf" | "body" | "tex">("pdf");
  useEffect(() => {
    if (!key && contexts.data?.items.length) setKey(contexts.data.items[0].key);
  }, [contexts.data, key]);
  const selected = useQuery({
    queryKey: ["context", key],
    queryFn: () =>
      api<Context & { jd_text?: string }>(
        `/api/context/get?key=${encodeURIComponent(key)}`,
      ),
    enabled: !!key,
  });
  const pdfUrl = result?.context_key
    ? `/api/context/pdf?key=${encodeURIComponent(result.context_key)}&document=cover_letter`
    : "";
  useEffect(
    () => setJd(selected.data?.jd_text || ""),
    [key, selected.data?.jd_text],
  );
  const generate = useMutation({
    mutationFn: async () => {
      if (!jd.trim())
        throw new Error("Paste the job description for this application.");
      const form = new FormData();
      form.set("jd_text", jd);
      form.set(
        "resume_tex",
        selected.data?.humanized_tex || selected.data?.resume_tex || "",
      );
      form.set("use_humanize", String(humanize));
      form.set("tone", tone);
      form.set("length", length);
      return api<CoverResult>("/api/coverletter", {
        method: "POST",
        body: form,
      });
    },
    onSuccess: (d) => {
      setResult(d);
      setBody(d.body_text || "");
      setView("body");
      notify("Cover letter created and saved");
    },
    onError: (e) => notify(e.message, "error"),
  });
  const compile = useMutation({
    mutationFn: async () => {
      if (!result) throw new Error("Generate a letter first.");
      const form = new FormData();
      form.set("tex_string", result.tex_string);
      form.set("body_text", body);
      form.set("company", result.company);
      form.set("role", result.role);
      form.set("tone", tone);
      form.set("context_key", result.context_key || "");
      return api<
        Pick<
          CoverResult,
          "tex_string" | "body_text" | "pdf_base64" | "word_count"
        >
      >("/api/coverletter/compile", { method: "POST", body: form });
    },
    onSuccess: (d) => {
      setResult((v) => (v ? { ...v, ...d } : v));
      setBody(d.body_text || "");
      setView("pdf");
      notify("Edited letter passed SuperHuman and saved to Desktop");
    },
    onError: (e) => notify(e.message, "error"),
  });
  return (
    <div className="page">
      <PageTitle
        eyebrow="Cover letters"
        title="Write a letter grounded in your résumé."
        description="Choose the exact application context, control the tone and length, and review the final document before downloading."
      />
      <div className="cover-grid">
        <aside className="panel padded cover-settings">
          <div className="section-head">
            <div>
              <h2>Letter setup</h2>
              <p>Every claim comes from the selected résumé.</p>
            </div>
          </div>
          <label className="field">
            <span>Application context</span>
            <select
              value={key}
              onChange={(e) => {
                setKey(e.target.value);
                setResult(null);
              }}
            >
              <option value="">Choose a saved résumé…</option>
              {contexts.data?.items.map((x) => (
                <option key={x.key} value={x.key}>
                  {x.company} · {x.role}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Job description</span>
            <textarea
              value={jd}
              onChange={(e) => setJd(e.target.value)}
              placeholder="Saved automatically when available. Paste the job description for older PDFs."
            />
          </label>
          <label className="field">
            <span>Tone</span>
            <select value={tone} onChange={(e) => setTone(e.target.value)}>
              <option value="balanced">Natural and balanced</option>
              <option value="formal">Professional and formal</option>
              <option value="conversational">Warm and conversational</option>
              <option value="confident">Direct and confident</option>
            </select>
          </label>
          <label className="field">
            <span>Length</span>
            <select value={length} onChange={(e) => setLength(e.target.value)}>
              <option value="concise">Concise</option>
              <option value="standard">Standard</option>
              <option value="detailed">Detailed</option>
            </select>
          </label>
          <label className="switch-row compact">
            <span>
              <b>Humanize body</b>
              <small>Simple, natural Indian English.</small>
            </span>
            <input
              type="checkbox"
              checked={humanize}
              onChange={(e) => setHumanize(e.target.checked)}
            />
          </label>
          <div className="context-card">
            <b>{selected.data?.company || "No context selected"}</b>
            <span>
              {selected.data?.role || "Choose a saved résumé to continue."}
            </span>
            <small>
              {jd.trim()
                ? "Job description ready"
                : "Paste job description to continue"}
            </small>
          </div>
          {generate.isError && (
            <div className="inline-error">{generate.error.message}</div>
          )}
          <button
            className="button primary cover-generate"
            disabled={
              !key ||
              !jd.trim() ||
              !selected.data?.resume_tex ||
              generate.isPending
            }
            onClick={() => generate.mutate()}
          >
            <FileText size={16} />
            {generate.isPending ? "Writing letter…" : "Generate cover letter"}
          </button>
        </aside>
        <section className="panel cover-document">
          <header className="document-head">
            <div>
              <span className="eyebrow">
                {result
                  ? `${body.trim().split(/\s+/).filter(Boolean).length || result.word_count} words · ${result.company_type || "tailored"}`
                  : "Document preview"}
              </span>
              <h2>{result ? result.company : "Your cover letter"}</h2>
              <p>
                {result
                  ? result.role
                  : "The finished document will appear here."}
              </p>
            </div>
            {result && (
              <div className="document-actions">
                {view === "body" ? (
                  <>
                    <button
                      className="button ghost"
                      onClick={() => {
                        navigator.clipboard.writeText(body);
                        notify("Letter copied");
                      }}
                    >
                      <Copy size={15} />
                      Copy letter
                    </button>
                    <button
                      className="button primary"
                      disabled={compile.isPending || !body.trim()}
                      onClick={() => compile.mutate()}
                    >
                      {compile.isPending ? "Compiling…" : "Recompile PDF"}
                    </button>
                  </>
                ) : view === "tex" ? (
                  <button
                    className="button ghost"
                    onClick={() =>
                      navigator.clipboard.writeText(result.tex_string)
                    }
                  >
                    <Copy size={15} />
                    Copy source
                  </button>
                ) : pdfUrl ? (
                  <a
                    className="button ghost"
                    href={pdfUrl}
                    download={`${result.company}-${result.role}-cover-letter.pdf`}
                  >
                    <Download size={15} />
                    Download PDF
                  </a>
                ) : null}
              </div>
            )}
          </header>
          {result && (
            <div className="document-toolbar">
              <div className="segmented">
                <button
                  className={view === "body" ? "active" : ""}
                  onClick={() => setView("body")}
                >
                  Readable letter
                </button>
                <button
                  className={view === "pdf" ? "active" : ""}
                  onClick={() => setView("pdf")}
                >
                  PDF preview
                </button>
                <button
                  className={view === "tex" ? "active" : ""}
                  onClick={() => setView("tex")}
                >
                  LaTeX source
                </button>
              </div>
              {view === "body" && (
                <span className="edit-note">
                  Edit the text, then recompile the PDF.
                </span>
              )}
            </div>
          )}
          <div className="document-canvas cover-canvas">
            {generate.isPending ? (
              <div className="progress-state">
                <span className="progress-ring" />
                <b>Creating a role-specific letter</b>
                <p>
                  Researching the company, mapping requirements, checking facts,
                  and compiling the final PDF.
                </p>
              </div>
            ) : result ? (
              view === "body" ? (
                <textarea
                  className="letter-editor"
                  value={body}
                  onChange={(e) => setBody(e.target.value)}
                />
              ) : view === "pdf" && pdfUrl ? (
                <iframe title="Cover letter preview" src={pdfUrl} />
              ) : (
                <pre>{result.tex_string}</pre>
              )
            ) : (
              <Empty
                title="No cover letter yet"
                copy="Choose an application context and generate your tailored letter."
              />
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function Preview() {
  const contexts = useQuery({
    queryKey: ["contexts", "preview"],
    queryFn: () => api<{ items: Context[] }>("/api/context/list?limit=5000"),
  });
  const archive = useQuery({
    queryKey: ["application-archive"],
    queryFn: fetchApplicationArchive,
    staleTime: 300000,
  });
  const [key, setKey] = useState(() => {
    const saved = localStorage.getItem("astra-preview-context") || "";
    return saved.startsWith("ctx:app:") ? saved.slice(4) : saved;
  });
  const [variant, setVariant] = useState<"humanized" | "optimized">(
    "optimized",
  );
  const [view, setView] = useState<"pdf" | "tex">("pdf");
  const [filter, setFilter] = useState(
    () => localStorage.getItem("astra-preview-filter") || "",
  );
  const [sort, setSort] = useState<"newest" | "company">("newest");
  const [fileIndex, setFileIndex] = useState(0);
  const [shown, setShown] = useState(120);
  useEffect(() => {
    if (key) return;
    const generated = contexts.data?.items.find(
      (x) => x.has_pdf && !x.key.startsWith("app:"),
    );
    if (generated) setKey(`ctx:${generated.key}`);
    else if (archive.data?.length) setKey(`app:${archive.data[0].id}`);
  }, [contexts.data, archive.data, key]);
  useEffect(() => {
    if (key) localStorage.setItem("astra-preview-context", key);
  }, [key]);
  useEffect(
    () => localStorage.setItem("astra-preview-filter", filter),
    [filter],
  );
  const isArchive = key.startsWith("app:");
  const contextKey = key.startsWith("ctx:") ? key.slice(4) : key;
  const detail = useQuery({
    queryKey: ["context", contextKey],
    queryFn: () =>
      api<Context>(`/api/context/get?key=${encodeURIComponent(contextKey)}`),
    enabled: !!key && !isArchive,
  });
  const archiveItem = isArchive
    ? archive.data?.find((x) => x.id === key.slice(4))
    : undefined;
  const archiveFiles = (archiveItem?.files || []).filter((f) =>
    f.path.toLowerCase().endsWith(".pdf"),
  );
  const selectedFile = archiveFiles[fileIndex];
  const item = detail.data;
  const hasHumanized = !!(
    item?.pdf_base64_humanized ||
    item?.humanized_tex ||
    item?.has_humanized_pdf
  );
  const activeVariant =
    variant === "humanized" && hasHumanized
      ? "humanized"
      : !item?.has_optimized_pdf && item?.has_humanized_pdf
        ? "humanized"
        : "optimized";
  const tex =
    activeVariant === "humanized" ? item?.humanized_tex : item?.resume_tex;
  const pdfUrl = item?.has_pdf
    ? `/api/context/pdf?key=${encodeURIComponent(contextKey)}&variant=${activeVariant}`
    : "";
  const archiveUrl = archiveItem
    ? `/api/applications/${archiveItem.id}/resume?file_index=${fileIndex}`
    : "";
  const documents = useMemo(() => {
    const recent = (contexts.data?.items || [])
      .filter((x) => x.has_pdf && !x.key.startsWith("app:"))
      .map((x) => ({
        id: `ctx:${x.key}`,
        company: x.company,
        role: x.role,
        date: x.updated_at,
        kind: "Generated",
      }));
    const seen = new Set(
      recent.map((x) => `${x.company}__${x.role}`.toLowerCase()),
    );
    const historical = (archive.data || [])
      .filter(
        (x) =>
          x.files?.some((f) => f.path.toLowerCase().endsWith(".pdf")) &&
          !seen.has(`${x.company}__${x.role}`.toLowerCase()),
      )
      .map((x) => ({
        id: `app:${x.id}`,
        company: x.company,
        role: x.role,
        date: x.applied_at,
        kind: "Application",
      }));
    return [...recent, ...historical];
  }, [contexts.data, archive.data]);
  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return documents
      .filter((x) => !q || `${x.company} ${x.role}`.toLowerCase().includes(q))
      .sort((a, b) =>
        sort === "company"
          ? a.company.localeCompare(b.company)
          : String(b.date || "").localeCompare(String(a.date || "")),
      );
  }, [documents, filter, sort]);
  const selectedPosition = visible.findIndex((x) => x.id === key);
  const moveSelection = (delta: number) => {
    const next = visible[selectedPosition + delta];
    if (next) {
      setKey(next.id);
      setFileIndex(0);
      setView("pdf");
    }
  };
  const download = (href: string, name: string) => {
    const link = document.createElement("a");
    link.href = href;
    link.download = name;
    link.click();
  };
  const safeName =
    `${item?.company || "ASTRA"}-${item?.role || "resume"}-${activeVariant}`.replace(
      /[^a-z0-9._-]+/gi,
      "-",
    );
  return (
    <div className="page preview-page">
      <PageTitle
        eyebrow="Document workspace"
        title="Review every detail before you send."
        description="Search generated résumés and the complete manual and automatic application archive."
        action={
          <a className="button primary" href="#/studio">
            <Command size={16} />
            Create new résumé
          </a>
        }
      />
      <div className="preview-grid">
        <aside className="panel preview-library">
          <div className="preview-library-head">
            <div>
              <h2>Résumé library</h2>
              <p>
                {documents.length.toLocaleString()} PDFs · grouped by
                application
              </p>
            </div>
            <button
              className="icon-button"
              onClick={() => {
                contexts.refetch();
                archive.refetch();
              }}
              title="Refresh library"
            >
              <RefreshCw size={16} />
            </button>
          </div>
          <label className="search preview-search">
            <Search size={16} />
            <input
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                setShown(120);
              }}
              placeholder="Search company or role"
            />
          </label>
          <div className="library-filters">
            <span>PDF only</span>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as "newest" | "company")}
            >
              <option value="newest">Newest first</option>
              <option value="company">Company A–Z</option>
            </select>
          </div>
          <div className="preview-list">
            {contexts.isLoading || archive.isLoading ? (
              <Skeleton rows={7} />
            ) : contexts.isError && archive.isError ? (
              <Empty
                title="Library unavailable"
                copy="Refresh after checking the ASTRA service."
              />
            ) : visible.length ? (
              <>
                {visible.slice(0, shown).map((x) => (
                  <button
                    key={x.id}
                    className={key === x.id ? "selected" : ""}
                    onClick={() => {
                      setKey(x.id);
                      setFileIndex(0);
                      setView("pdf");
                    }}
                  >
                    <span className="company-mark">
                      {(x.company || "A").slice(0, 2).toUpperCase()}
                    </span>
                    <span>
                      <b>{x.company || "Untitled company"}</b>
                      <small>
                        {x.role || "Résumé"} · {x.kind}
                      </small>
                      <time>{fmt(x.date)}</time>
                    </span>
                    {key === x.id && <Check size={15} />}
                  </button>
                ))}
                {shown < visible.length && (
                  <button
                    className="load-more"
                    onClick={() => setShown((v) => v + 120)}
                  >
                    Load more · {(visible.length - shown).toLocaleString()}{" "}
                    remaining
                  </button>
                )}
              </>
            ) : (
              <Empty
                title="No résumés found"
                copy="Try another company or role."
              />
            )}
          </div>
        </aside>
        <section className="panel preview-workspace">
          {!key ? (
            <Empty
              title="No résumé selected"
              copy="Select a saved document or create a new tailored résumé."
            />
          ) : isArchive && archiveItem ? (
            <>
              <header className="document-head">
                <div>
                  <span className="eyebrow">
                    Application archive · {archiveFiles.length} version
                    {archiveFiles.length === 1 ? "" : "s"}
                  </span>
                  <h2>{archiveItem.company}</h2>
                  <p>
                    {archiveItem.role} · Applied {fmt(archiveItem.applied_at)}
                  </p>
                </div>
                <div className="document-actions">
                  <button
                    className="icon-button"
                    disabled={selectedPosition <= 0}
                    onClick={() => moveSelection(-1)}
                    title="Previous résumé"
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <button
                    className="icon-button"
                    disabled={
                      selectedPosition < 0 ||
                      selectedPosition >= visible.length - 1
                    }
                    onClick={() => moveSelection(1)}
                    title="Next résumé"
                  >
                    <ChevronRight size={16} />
                  </button>
                  <button
                    className="button ghost"
                    onClick={() =>
                      api(
                        `/api/applications/${archiveItem.id}/reveal?file_index=${fileIndex}`,
                        { method: "POST" },
                      )
                        .then(() => notify("Opened in Finder"))
                        .catch((e) => notify(e.message, "error"))
                    }
                  >
                    <FolderOpen size={15} />
                    Finder
                  </button>
                  <a className="button ghost" href={archiveUrl} target="_blank">
                    <ExternalLink size={15} />
                    Open
                  </a>
                  <a
                    className="button ghost"
                    href="#/applications"
                    onClick={() =>
                      localStorage.setItem(
                        "astra-focus-application",
                        JSON.stringify({
                          id: archiveItem.id,
                          company: archiveItem.company,
                        }),
                      )
                    }
                  >
                    Application
                  </a>
                </div>
              </header>
              <div className="document-toolbar">
                <label className="version-picker">
                  <span>Version</span>
                  <select
                    value={fileIndex}
                    onChange={(e) => setFileIndex(Number(e.target.value))}
                  >
                    {archiveFiles.map((f, i) => (
                      <option value={i} key={f.path}>
                        {i === 0 ? "Current" : "Version " + (i + 1)} ·{" "}
                        {f.path.split("/").pop()}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="file-meta">
                  <span>
                    {selectedFile?.size
                      ? `${Math.round(selectedFile.size / 1024)} KB`
                      : "PDF"}
                  </span>
                  <span>{fmt(selectedFile?.modified_at)}</span>
                </div>
                <button
                  className="icon-button fullscreen"
                  onClick={() =>
                    document
                      .querySelector<HTMLElement>(".document-canvas")
                      ?.requestFullscreen()
                  }
                  title="Full screen"
                >
                  <Maximize2 size={16} />
                </button>
              </div>
              <div className="document-canvas">
                <iframe
                  key={archiveUrl}
                  title={`${archiveItem.company} résumé`}
                  src={archiveUrl}
                />
              </div>
            </>
          ) : detail.isLoading ? (
            <Skeleton rows={8} />
          ) : detail.isError ? (
            <Empty
              title="Résumé unavailable"
              copy="This saved context could not be opened."
              action={
                <button
                  className="button ghost"
                  onClick={() => detail.refetch()}
                >
                  <RefreshCw size={15} />
                  Try again
                </button>
              }
            />
          ) : (
            <>
              <header className="document-head">
                <div>
                  <span className="eyebrow">{activeVariant} version</span>
                  <h2>{item?.company}</h2>
                  <p>
                    {item?.role} · Updated {fmt(item?.updated_at)}
                  </p>
                </div>
                <div className="document-actions">
                  <button
                    className="icon-button"
                    disabled={selectedPosition <= 0}
                    onClick={() => moveSelection(-1)}
                    title="Previous résumé"
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <button
                    className="icon-button"
                    disabled={
                      selectedPosition < 0 ||
                      selectedPosition >= visible.length - 1
                    }
                    onClick={() => moveSelection(1)}
                    title="Next résumé"
                  >
                    <ChevronRight size={16} />
                  </button>
                  {view === "tex" && tex ? (
                    <button
                      className="button ghost"
                      onClick={() => {
                        navigator.clipboard.writeText(tex);
                        notify("LaTeX copied");
                      }}
                    >
                      <Copy size={15} />
                      Copy source
                    </button>
                  ) : null}
                  {view === "pdf" && pdfUrl ? (
                    <button
                      className="button ghost"
                      onClick={() => download(pdfUrl, `${safeName}.pdf`)}
                    >
                      <Download size={15} />
                      Download PDF
                    </button>
                  ) : null}
                  {view === "tex" && tex ? (
                    <button
                      className="button ghost"
                      onClick={() =>
                        download(
                          URL.createObjectURL(
                            new Blob([tex], { type: "text/plain" }),
                          ),
                          `${safeName}.tex`,
                        )
                      }
                    >
                      <Download size={15} />
                      Download .tex
                    </button>
                  ) : null}
                </div>
              </header>
              <div className="document-toolbar">
                <div className="segmented">
                  <button
                    className={activeVariant === "optimized" ? "active" : ""}
                    onClick={() => setVariant("optimized")}
                  >
                    Optimized
                  </button>
                  <button
                    className={activeVariant === "humanized" ? "active" : ""}
                    disabled={!hasHumanized}
                    onClick={() => setVariant("humanized")}
                  >
                    Humanized
                  </button>
                </div>
                <div className="segmented">
                  <button
                    className={view === "pdf" ? "active" : ""}
                    onClick={() => setView("pdf")}
                  >
                    PDF preview
                  </button>
                  <button
                    className={view === "tex" ? "active" : ""}
                    onClick={() => setView("tex")}
                  >
                    LaTeX source
                  </button>
                </div>
                {item?.fit_score !== undefined && item.fit_score !== "" ? (
                  <div className="fit-score">
                    <span>Fit score</span>
                    <b>{String(item.fit_score).replace(/%$/g, "")}%</b>
                  </div>
                ) : null}
                <button
                  className="icon-button fullscreen"
                  onClick={() =>
                    document
                      .querySelector<HTMLElement>(".document-canvas")
                      ?.requestFullscreen()
                  }
                  title="Full screen"
                >
                  <Maximize2 size={16} />
                </button>
              </div>
              <div className="document-canvas">
                {view === "pdf" ? (
                  pdfUrl ? (
                    <iframe title={`${item?.company} résumé`} src={pdfUrl} />
                  ) : (
                    <Empty
                      title="PDF is not available"
                      copy="The source is saved, but this version has no compiled PDF."
                    />
                  )
                ) : tex ? (
                  <pre>{tex}</pre>
                ) : (
                  <Empty
                    title="Source is not available"
                    copy="This résumé does not contain LaTeX source."
                  />
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

function Legacy({ route, dark }: { route: Route; dark: boolean }) {
  const meta = routes.find((x) => x.id === route)!;
  const src = `${legacy[route]}&theme=${dark ? "dark" : "light"}`;
  return (
    <div className="legacy">
      <div className="legacy-label">
        <span>{meta.label}</span>
        <small>Existing workflow · React migration in progress</small>
      </div>
      <iframe key={src} title={meta.label} src={src} />
    </div>
  );
}

function GlobalSearch({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [stage, setStage] = useState("all");
  const [sort, setSort] = useState<"newest" | "company">("newest");
  const archive = useQuery({
    queryKey: ["application-archive"],
    queryFn: fetchApplicationArchive,
    staleTime: 300000,
    enabled: open,
  });
  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);
  const pdfCount = (archive.data || []).filter((x) =>
    x.files?.some((f) => f.path.toLowerCase().endsWith(".pdf")),
  ).length;
  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return (archive.data || [])
      .filter(
        (x) =>
          x.files?.some((f) => f.path.toLowerCase().endsWith(".pdf")) &&
          (stage === "all" || x.status === stage) &&
          `${x.company} ${x.role} ${x.status}`.toLowerCase().includes(q),
      )
      .sort((a, b) =>
        sort === "company"
          ? a.company.localeCompare(b.company)
          : String(b.applied_at || "").localeCompare(
              String(a.applied_at || ""),
            ),
      )
      .slice(0, 30);
  }, [archive.data, query, stage, sort]);
  const choose = (item: Application) => {
    localStorage.setItem("astra-preview-context", `app:${item.id}`);
    location.hash = "#/preview";
    onClose();
  };
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content
          className="search-dialog"
          aria-describedby="search-description"
        >
          <Dialog.Title className="sr-only">Search ASTRA</Dialog.Title>
          <Dialog.Description id="search-description" className="sr-only">
            Search applications and resume documents
          </Dialog.Description>
          <div className="search-dialog-input">
            <Search size={20} />
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Search ${pdfCount.toLocaleString()} PDF résumés…`}
            />
            <button onClick={onClose}>
              <X size={17} />
            </button>
          </div>
          <div className="search-filterbar">
            <span>PDF only</span>
            <select value={stage} onChange={(e) => setStage(e.target.value)}>
              <option value="all">All stages</option>
              <option value="applied">Applied</option>
              <option value="recruiter_call">Recruiter call</option>
              <option value="interview">Interview</option>
              <option value="offer">Offer</option>
            </select>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as "newest" | "company")}
            >
              <option value="newest">Newest</option>
              <option value="company">Company A–Z</option>
            </select>
          </div>
          <div className="search-results">
            {archive.isLoading ? (
              <Skeleton rows={6} />
            ) : !query ? (
              <div className="search-hint">
                <Command size={21} />
                <b>Find anything in your workspace</b>
                <span>Search by company, role, or application stage.</span>
              </div>
            ) : results.length ? (
              results.map((x) => (
                <button key={x.id} onClick={() => choose(x)}>
                  <span className="company-mark">
                    {x.company.slice(0, 2).toUpperCase()}
                  </span>
                  <span>
                    <b>{x.company}</b>
                    <small>{x.role}</small>
                  </span>
                  <em className={`status status-${x.status}`}>
                    {x.status.replace("_", " ")}
                  </em>
                  <ChevronRight size={15} />
                </button>
              ))
            ) : (
              <Empty
                title="No matching records"
                copy="Try a company name or a broader role title."
              />
            )}
          </div>
          <footer>
            {results.length} results · Esc to close · PDF résumés only
          </footer>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);
  const [menu, setMenu] = useState(false);
  const [search, setSearch] = useState(false);
  const [dark, setDark] = useState(
    () => localStorage.getItem("astra-theme") === "dark",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("astra-theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => {
    const fn = () => {
      setRoute(routeFromHash());
      setMenu(false);
    };
    addEventListener("hashchange", fn);
    return () => removeEventListener("hashchange", fn);
  }, []);
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearch(true);
      }
    };
    addEventListener("keydown", fn);
    return () => removeEventListener("keydown", fn);
  }, []);
  const page = useMemo(
    () =>
      route === "studio" ? (
        <ResumeStudio />
      ) : route === "jobright" ? (
        <JobrightJobs />
      ) : route === "internet" ? (
        <InternetJobs />
      ) : route === "decisions" ? (
        <DecisionCentre />
      ) : route === "operations" ? (
        <AgentOperations />
      ) : route === "applications" ? (
        <Applications />
      ) : route === "genie" ? (
        <Genie />
      ) : route === "talk" ? (
        <Talk />
      ) : route === "preview" ? (
        <Preview />
      ) : route === "cover" ? (
        <CoverLetters />
      ) : route === "mastermind" ? (
        <MasterMind />
      ) : route === "superhuman" ? (
        <SuperHuman />
      ) : (
        <Legacy route={route} dark={dark} />
      ),
    [route, dark],
  );
  return (
    <div className="app">
      <Sidebar route={route} open={menu} onClose={() => setMenu(false)} />
      <div className="workspace">
        <Header
          route={route}
          onMenu={() => setMenu(true)}
          dark={dark}
          onTheme={() => setDark((v) => !v)}
          onSearch={() => setSearch(true)}
        />
        <main>
          <AnimatePresence mode="wait">
            <motion.div
              key={route}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
            >
              {page}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      {menu && (
        <button
          className="scrim"
          onClick={() => setMenu(false)}
          aria-label="Close navigation"
        />
      )}
      <GlobalSearch open={search} onClose={() => setSearch(false)} />
      <ToastViewport />
    </div>
  );
}
