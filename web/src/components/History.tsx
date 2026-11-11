import { lazy, Suspense, useEffect, useState } from "react";
import { api, type Spec, type StudyRecord, type StudySummary, type TableRow } from "../api";
import ResultsTable from "./ResultsTable";
import Fields from "./Fields";

const Chart = lazy(() => import("./Chart"));

// Earlier studies, from the catalogue, and the comparison of studies that asked
// the same question. Kept below the four steps: it is somewhere to look back,
// not another step to take.

const SHOWN = ["case", "sweeps", "converged", "wall_time", "rel_flux_diff", "runner"];

function compact(rows: TableRow[], swept: string[]): TableRow[] {
  const columns = [...SHOWN.slice(0, 1), ...swept.map((p) => `param_${p}`), ...SHOWN.slice(1)];
  const present = columns.filter((c) => rows.some((r) => r[c] !== null && r[c] !== undefined));
  return rows.map((row) => Object.fromEntries(present.map((c) => [c.replace(/^param_/, ""), row[c] ?? null])));
}

function when(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function Charts({ specs }: { specs: Spec[] }) {
  if (!specs.length) return null;
  return (
    <Suspense fallback={<p className="hint">Drawing charts…</p>}>
      <div className="charts">
        {specs.map((spec, index) => (
          <Chart key={index} spec={spec} />
        ))}
      </div>
    </Suspense>
  );
}

/** Earlier studies of the same question, overlaid. Shown only when there are some. */
export function Compare({ group, studyId }: { group: string; studyId?: string | null }) {
  const [earlier, setEarlier] = useState<StudySummary[]>([]);
  const [charts, setCharts] = useState<Spec[] | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setCharts(null);
    api.compare(group)
      .then((r) => setEarlier(r.studies.filter((s) => s.id !== studyId)))
      .catch(() => setEarlier([]));
  }, [group, studyId]);

  if (!earlier.length) return null;
  return (
    <div className="compare">
      {!charts ? (
        <button
          className="quiet"
          disabled={busy}
          onClick={() => {
            setBusy(true);
            api.compare(group).then((r) => setCharts(r.charts)).finally(() => setBusy(false));
          }}
        >
          Compare with {earlier.length} earlier {earlier.length === 1 ? "study" : "studies"} of this kind
        </button>
      ) : (
        <>
          <p className="examples-head">Studies varying the same parameters, overlaid</p>
          {charts.length ? <Charts specs={charts} /> : <p className="hint">Nothing in common to chart.</p>}
        </>
      )}
    </div>
  );
}

export default function History({ refresh }: { refresh: unknown }) {
  const [studies, setStudies] = useState<StudySummary[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [record, setRecord] = useState<StudyRecord | null>(null);
  const [charts, setCharts] = useState<Spec[]>([]);

  useEffect(() => {
    api.studies().then((r) => setStudies(r.studies)).catch(() => setStudies([]));
  }, [refresh]);

  useEffect(() => {
    setRecord(null);
    setCharts([]);
    if (!open) return;
    api.study(open).then(setRecord).catch(() => setRecord(null));
    api.studyCharts(open).then((r) => setCharts(r.charts)).catch(() => setCharts([]));
  }, [open]);

  if (!studies || studies.length === 0) return null;
  return (
    <section className="history">
      <h2 className="history-title">Earlier studies</h2>
      <ul className="history-list">
        {studies.map((s) => (
          <li key={s.id}>
            <button
              className={`history-item ${open === s.id ? "open" : ""}`}
              onClick={() => setOpen(open === s.id ? null : s.id)}
              aria-expanded={open === s.id}
            >
              <span className="history-name">{s.name}</span>
              <span className="history-meta">
                {s.template} · {s.cases} case{s.cases === 1 ? "" : "s"}
                {s.converged_cases !== null && s.converged_cases < s.cases ? ` · ${s.cases - s.converged_cases} not converged` : ""}
                {" · "}{s.runner}{" · "}{when(s.created_at)}
              </span>
            </button>
            {open === s.id && (
              <div className="history-body">
                {!record ? (
                  <p className="hint">Loading…</p>
                ) : (
                  <>
                    {record.engine_note && <div className="notice plain">{record.engine_note}</div>}
                    <ResultsTable rows={compact(record.cases.map((c) => c.row), record.swept)} />
                    <Charts specs={charts} />
                    <Fields studyId={record.id} />
                    <Compare group={record.group_key} studyId={record.id} />
                  </>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
