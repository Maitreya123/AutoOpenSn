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

/** An edited spec, as the server read it. */
export type SpecCheck =
  | { ok: true; spec_yaml: string | null; spec_json: string | null; cases: number; error?: undefined }
  | { ok: false; error: string; spec_yaml?: undefined; spec_json?: undefined; cases?: undefined };

export type SpecDraft = {
  ok: boolean;
  summary: string;
  spec_yaml: string | null;
  spec_json: string | null;
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
  study_id?: string | null;
  group?: string | null;
  /** Vega-Lite specifications, worked out by the server from the study. */
  charts?: Record<string, unknown>[] | null;
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

export type FieldFile = {
  path: string;
  kind: "vtk" | "profile";
  arrays: string[];
  columns: string[];
  pieces: number;
  groups: number;
  default_array: string | null;
};

export type CaseFields = { case_id: string; label: string; fields: FieldFile[] };

export type StudySummary = {
  id: string;
  name: string;
  template: string;
  group_key: string;
  swept: string[];
  runner: string;
  engine_note: string | null;
  created_at: string;
  cases: number;
  ok_cases: number;
  converged_cases: number | null;
  cache_hits: number;
  wall_time_total: number | null;
};

export type StudyRecord = Omit<StudySummary, "cases"> & {
  spec_yaml: string;
  cases: { case_id: string; label: string; is_reference: boolean; parameters: Record<string, unknown>; row: TableRow }[];
};

export type Spec = Record<string, unknown>;

export const api = {
  health: () => request<Health>("/api/health"),

  generateSpec: (prompt: string) =>
    request<SpecDraft>("/api/spec", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),

  checkSpec: (specYaml: string) =>
    request<SpecCheck>("/api/spec/check", {
      method: "POST",
      body: JSON.stringify({ spec_yaml: specYaml }),
    }),

  startRun: (specYaml: string) =>
    request<StartedRun>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ spec_yaml: specYaml, runner: "auto" }),
    }),

  job: (id: string) => request<Job>(`/api/runs/${id}`),

  studies: (limit = 12) => request<{ studies: StudySummary[] }>(`/api/studies?limit=${limit}`),

  study: (id: string) => request<StudyRecord>(`/api/studies/${encodeURIComponent(id)}`),

  studyCharts: (id: string) =>
    request<{ charts: Spec[] }>(`/api/studies/${encodeURIComponent(id)}/charts`),

  compare: (group: string) =>
    request<{ studies: StudySummary[]; charts: Spec[] }>(`/api/compare?group=${encodeURIComponent(group)}`),

  fields: (id: string) =>
    request<{ cases: CaseFields[] }>(`/api/studies/${encodeURIComponent(id)}/fields`),

  fieldChart: (id: string, caseId: string, file: string) =>
    request<{ chart: Spec | null }>(
      `/api/studies/${encodeURIComponent(id)}/fields/chart?case=${encodeURIComponent(caseId)}&file=${encodeURIComponent(file)}`,
    ),

  /** A URL, not a fetch: the browser loads and caches the PNG itself. */
  fieldImage: (id: string, caseId: string, file: string, array: string | null) =>
    `/api/studies/${encodeURIComponent(id)}/fields/image?case=${encodeURIComponent(caseId)}` +
    `&file=${encodeURIComponent(file)}${array ? `&array=${encodeURIComponent(array)}` : ""}`,

  /** With a study id, the server also describes that study's charts and fields to the model. */
  explain: (table: TableRow[], specYaml: string | null, studyId: string | null = null) =>
    request<Narrative>("/api/explain", {
      method: "POST",
      body: JSON.stringify({ table, spec_yaml: specYaml, study_id: studyId }),
    }),

  chat: (
    table: TableRow[],
    messages: { role: string; content: string }[],
    specYaml: string | null,
    studyId: string | null = null,
  ) =>
    request<Narrative>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ table, messages, spec_yaml: specYaml, study_id: studyId }),
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
