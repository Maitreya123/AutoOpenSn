// The HTTP surface, typed once so the components never build a URL themselves.
//
// Every call goes through `request`, which turns a non-2xx into an Error
// carrying the API's own `detail` string. That matters more here than in most
// apps: the useful errors in this tool are domain errors — a tolerance outside
// its declared range, a cluster that cannot be reached — and the API already
// phrases those well. Replacing them with "Request failed" would throw away the
// only part of the message worth reading.

export type Scenario = {
  name: string;
  title: string;
  summary: string;
  section: string;
  parameters: string[];
  sweepable: boolean;
  has_gold: boolean;
};

export type Citation = { number: number; location: string; title: string; text: string };

export type SpecDraft = {
  ok: boolean;
  summary: string;
  spec_yaml: string | null;
  cases: number;
  candidates: Scenario[];
  citations: Citation[];
  notes: string[];
  errors: string[];
  attempts: string[];
  revisions: number;
  review_status: string;
  model: string | null;
  elapsed: number;
};

export type TableRow = Record<string, string | number | boolean | null>;

export type Job = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  finished_at: string | null;
  progress: string[];
  total: number;
  completed: number;
  error: string | null;
  summary: string | null;
  run_root: string | null;
  cache_hits: string[];
  label: string;
  note: string;
  table?: TableRow[] | null;
};

export type StartedRun = {
  id: string;
  status: string;
  total: number;
  runner: string;
  note: string;
};

export type Narrative = {
  text: string;
  warnings?: string[];
  unsupported?: string[];
  model: string | null;
  elapsed: number;
};

export type Health = {
  version: string;
  pack_commit: string;
  llm_available: boolean;
  knowledge_pack_available: boolean;
  scenarios: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // A non-JSON error body is rare and not more informative than the status.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),

  generateSpec: (prompt: string) =>
    request<SpecDraft>("/api/spec", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),

  startRun: (specYaml: string) =>
    request<StartedRun>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ spec_yaml: specYaml, runner: "auto" }),
    }),

  job: (id: string) => request<Job>(`/api/runs/${id}`),

  explain: (table: TableRow[], specYaml: string | null) =>
    request<Narrative>("/api/explain", {
      method: "POST",
      body: JSON.stringify({ table, spec_yaml: specYaml }),
    }),

  chat: (table: TableRow[], messages: { role: string; content: string }[], specYaml: string | null) =>
    request<Narrative>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ table, messages, spec_yaml: specYaml }),
    }),
};

// Poll until the job settles. The interval is deliberately not adaptive: a
// study is seconds in the reference solver and minutes on a cluster, and a
// fixed second costs nothing against either.
export async function pollJob(
  id: string,
  onUpdate: (job: Job) => void,
  intervalMs = 1000,
): Promise<Job> {
  for (;;) {
    const job = await api.job(id);
    onUpdate(job);
    if (job.status === "succeeded" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
