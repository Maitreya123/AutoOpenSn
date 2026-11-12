import { lazy, Suspense, useEffect, useState } from "react";
import { api, pollJob, type Job, type Narrative, type SpecCheck, type SpecDraft, type TableRow } from "./api";
import Step from "./components/Step";
import ResultsTable from "./components/ResultsTable";
// Loaded on demand: Vega is most of the bundle, and charts appear only after a
// study finishes, so there is no reason to make the first page wait for it.
const Chart = lazy(() => import("./components/Chart"));
import Chat, { type Turn } from "./components/Chat";
import Examples from "./components/Examples";
import Masthead from "./components/Masthead";
import Fields from "./components/Fields";
import History, { Compare } from "./components/History";

// Four steps, in order, and nothing to configure.
//
// The engine behind this has runners, caches, fixture directories, retry
// budgets and a reviewer. None of that appears here. A person describing a
// physics study should not have to know what a fixture replay is, and the API
// picks the best available runner itself. The one thing that is never hidden is
// *which* engine produced the numbers, because a results table that does not
// say whether it came from OpenSn or from this package's own solver is a table
// nobody should quote.

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [draft, setDraft] = useState<SpecDraft | null>(null);
  const [specError, setSpecError] = useState<string | null>(null);

  // The spec as the person has it in the editor, and what the server made of
  // it. Every edit is validated by the same code that would run it, so "Run
  // this" is only offered for a spec that will run.
  const [specText, setSpecText] = useState("");
  const [checked, setChecked] = useState<SpecCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const [ranSpec, setRanSpec] = useState<string | null>(null);

  const [job, setJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [narrative, setNarrative] = useState<Narrative | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [thinking, setThinking] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  // A question the provider has not seen takes twenty seconds to a minute;
  // repeating one takes about a second, because the gateway serves it from
  // cache. The slow case is the normal one for a person actually using this,
  // so the clock is shown and the wait is named rather than hidden.
  useEffect(() => {
    if (!generating) return;
    setElapsed(0);
    const started = Date.now();
    const tick = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(tick);
  }, [generating]);

  useEffect(() => {
    if (!draft?.ok || !draft.spec_yaml) return;
    setSpecText(draft.spec_yaml);
    setChecked({ ok: true, spec_yaml: draft.spec_yaml, spec_json: draft.spec_json, cases: draft.cases });
  }, [draft]);

  const edited = !!draft?.spec_yaml && specText !== draft.spec_yaml;

  useEffect(() => {
    if (!edited) return;
    setChecking(true);
    let current = true;
    const wait = setTimeout(() => {
      api.checkSpec(specText)
        .then((result) => current && setChecked(result))
        .catch((error) => current && setChecked({ ok: false, error: String(error) }))
        .finally(() => current && setChecking(false));
    }, 450);
    return () => {
      current = false;
      clearTimeout(wait);
    };
  }, [specText, edited]);

  const runnable = !!checked?.ok && !checking;
  // What the narration is told the study was: the spec that actually ran.
  const studySpec = ranSpec ?? draft?.spec_yaml ?? null;

  const table: TableRow[] = job?.table ?? [];
  const haveResults = job?.status === "succeeded" && table.length > 0;

  async function generate() {
    setGenerating(true);
    setSpecError(null);
    // Everything downstream describes the previous question, so it goes.
    setDraft(null);
    setRanSpec(null);
    setJob(null);
    setNarrative(null);
    setTurns([]);
    try {
      setDraft(await api.generateSpec(prompt.trim()));
    } catch (error) {
      setSpecError(error instanceof Error ? error.message : String(error));
    } finally {
      setGenerating(false);
    }
  }

  async function run() {
    if (!checked?.ok || !checked.spec_yaml) return;
    const spec = checked.spec_yaml;
    setRanSpec(spec);
    setRunning(true);
    setRunError(null);
    setNarrative(null);
    setTurns([]);
    try {
      const started = await api.startRun(spec);
      // Seed from the start response so the engine notice is on screen before
      // the first poll returns, rather than appearing a second later.
      setJob({
        id: started.id,
        status: "running",
        created_at: new Date().toISOString(),
        finished_at: null,
        progress: [],
        total: started.total,
        completed: 0,
        error: null,
        summary: null,
        run_root: null,
        cache_hits: [],
        label: "",
        note: started.note,
      });
      await pollJob(started.id, setJob);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  }

  async function analyse() {
    if (!haveResults) return;
    setThinking(true);
    setChatError(null);
    try {
      const result = await api.explain(table, studySpec, job?.study_id ?? null);
      setNarrative(result);
      setTurns([
        {
          role: "assistant",
          content: result.text,
          warnings: result.warnings,
        },
      ]);
    } catch (error) {
      setChatError(error instanceof Error ? error.message : String(error));
    } finally {
      setThinking(false);
    }
  }

  async function ask(question: string) {
    const asked: Turn[] = [...turns, { role: "user", content: question }];
    setTurns(asked);
    setThinking(true);
    setChatError(null);
    try {
      const answer = await api.chat(
        table,
        asked.map(({ role, content }) => ({ role, content })),
        studySpec,
        job?.study_id ?? null,
      );
      setTurns([
        ...asked,
        { role: "assistant", content: answer.text, warnings: answer.warnings },
      ]);
    } catch (error) {
      setChatError(error instanceof Error ? error.message : String(error));
    } finally {
      setThinking(false);
    }
  }

  // Both formats are the same spec. YAML is what the engine reads and what
  // `autoopensn run` takes; JSON is for anything that would rather parse it
  // than run it. Neither is converted here — the API serialises the one
  // validated object twice, so the two cannot drift apart.
  //
  // The YAML download is the editor's text as typed, comments and all; the JSON
  // is the server's reading of it.
  function downloadSpec(format: "yaml" | "json") {
    if (!checked?.ok) return;
    const text = format === "yaml" ? specText : checked.spec_json;
    if (!text) return;
    const blob = new Blob([text], {
      type: format === "yaml" ? "text/yaml" : "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${nameOf(checked.spec_yaml ?? specText)}.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const percent = job?.total ? Math.round((job.completed / job.total) * 100) : 0;

  return (
    <div className="shell">
      <Masthead />

      {/* 1 --------------------------------------------------------------- */}
      <Step number={1} title="What do you want to find out?" state={draft?.ok ? "done" : "active"}>
        <textarea
          rows={3}
          value={prompt}
          placeholder="e.g. compare GMRES tolerances 1e-4, 1e-6 and 1e-8 on the 1D transport problem"
          onChange={(event) => setPrompt(event.target.value)}
          disabled={generating}
        />
        <div className="row">
          <button onClick={generate} disabled={generating || !prompt.trim()}>
            {generating ? (
              <>
                <span className="spinner" />
                Writing the spec… {elapsed}s
              </>
            ) : (
              "Generate spec"
            )}
          </button>
        </div>
        {generating && elapsed >= 4 && (
          <p className="hint" style={{ marginTop: 10 }}>
            A question the model has not seen before usually takes twenty
            seconds to a minute. Repeating one is near-instant.
          </p>
        )}
        {!draft?.ok && !generating && (
          <Examples onPick={setPrompt} disabled={generating} />
        )}
        {specError && <div className="notice bad">{specError}</div>}
        {draft && !draft.ok && (
          <div className="notice bad">
            No valid spec could be written for that. {draft.errors.join(" ")}
          </div>
        )}
      </Step>

      {/* 2 --------------------------------------------------------------- */}
      <Step
        number={2}
        title="The input spec"
        state={!draft?.ok ? "waiting" : haveResults ? "done" : "active"}
        aside={
          draft?.ok
            ? edited
              ? checked?.ok
                ? `${checked.cases} run${checked.cases === 1 ? "" : "s"} · edited`
                : "edited"
              : `${draft.cases} run${draft.cases === 1 ? "" : "s"} · ${draft.summary} · ${draft.elapsed.toFixed(1)}s`
            : undefined
        }
      >
        {!draft?.ok ? (
          <p className="hint">Appears once a spec has been written.</p>
        ) : (
          <>
            <textarea
              className="spec spec-edit"
              value={specText}
              onChange={(event) => setSpecText(event.target.value)}
              spellCheck={false}
              rows={Math.min(18, Math.max(8, specText.split("\n").length + 1))}
              aria-label="The input spec, editable"
              disabled={running}
            />
            <div className="spec-status">
              {!edited ? (
                <span className="hint">Edit it directly if you want to change anything; it is checked as you type.</span>
              ) : checking ? (
                <span className="hint"><span className="spinner" />Checking…</span>
              ) : checked?.ok ? (
                <span className="spec-ok">✓ Valid · {checked.cases} run{checked.cases === 1 ? "" : "s"}</span>
              ) : (
                <span className="spec-bad">{checked?.error}</span>
              )}
              {edited && (
                <button className="link" onClick={() => setSpecText(draft.spec_yaml ?? "")}>
                  Undo my edits
                </button>
              )}
            </div>
            {/* The reviewer's points in one place, closed by default: advice
                about the spec, not a problem with it. */}
            {draft.notes.length > 0 && !edited && (
              <details className="notice plain review">
                <summary>
                  {draft.review_status === "revised"
                    ? `The domain reviewer suggested ${draft.notes.length === 1 ? "a change" : "changes"}; the spec above includes them`
                    : `The domain reviewer raised ${draft.notes.length} point${draft.notes.length === 1 ? "" : "s"}; the spec was left unchanged`}
                </summary>
                <ul>
                  {draft.notes.map((note, index) => (
                    <li key={index}>{note}</li>
                  ))}
                </ul>
              </details>
            )}
            <div className="row">
              <button onClick={run} disabled={running || !runnable}>
                {running ? <><span className="spinner" />Running…</> : "Run this"}
              </button>
              <button className="quiet" onClick={() => downloadSpec("yaml")} disabled={!runnable}>
                Download .yaml
              </button>
              <button className="quiet" onClick={() => downloadSpec("json")} disabled={!runnable}>
                Download .json
              </button>
            </div>
          </>
        )}
      </Step>

      {/* 3 --------------------------------------------------------------- */}
      <Step
        number={3}
        title="Results"
        state={!job ? "waiting" : haveResults ? "done" : "active"}
        aside={job?.summary ?? undefined}
      >
        {!job ? (
          <p className="hint">Appears once the spec has been run.</p>
        ) : (
          <>
            {job.note && <div className="notice plain">{job.note}</div>}

            {job.status === "running" || job.status === "queued" ? (
              <>
                <div className="bar">
                  <span style={{ width: `${percent}%` }} />
                </div>
                <p className="hint">
                  {job.completed} of {job.total} cases
                </p>
                {job.progress.length > 0 && (
                  <pre className="log">{job.progress.slice(-12).join("\n")}</pre>
                )}
              </>
            ) : null}

            {job.status === "failed" && (
              <div className="notice bad">{job.error ?? "The run failed."}</div>
            )}
            {runError && <div className="notice bad">{runError}</div>}

            {haveResults && (
              <>
                <ResultsTable rows={table} />
                {(job.charts ?? []).length > 0 && (
                  <Suspense fallback={<p className="hint">Drawing charts…</p>}>
                    <div className="charts">
                      {(job.charts ?? []).map((spec, index) => (
                        <Chart key={index} spec={spec} />
                      ))}
                    </div>
                  </Suspense>
                )}
                {job.study_id && <Fields studyId={job.study_id} />}
                {job.study_id && job.group && <Compare group={job.group} studyId={job.study_id} />}
                {job.cache_hits.length > 0 && (
                  <p className="hint" style={{ marginTop: 10 }}>
                    {job.cache_hits.length} case
                    {job.cache_hits.length === 1 ? " was" : "s were"} served from the
                    cache rather than rerun.
                  </p>
                )}
                {!narrative && (
                  <div className="row">
                    <button onClick={analyse} disabled={thinking}>
                      {thinking ? <><span className="spinner" />Reading…</> : "Analyse these results"}
                    </button>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </Step>

      {/* 4 --------------------------------------------------------------- */}
      <Step number={4} title="What the numbers show" state={narrative ? "active" : "waiting"}>
        {!narrative ? (
          <p className="hint">Appears once you analyse the results.</p>
        ) : (
          <>
            <Chat turns={turns} busy={thinking} onAsk={ask} />
            {chatError && <div className="notice bad">{chatError}</div>}
          </>
        )}
      </Step>
      <History refresh={job?.status === "succeeded" ? job.id : null} />
    </div>
  );
}

// The spec's own name, for the downloaded filename. Read off the YAML rather
// than tracked separately so an edited spec still saves under what it says it
// is.
function nameOf(specYaml: string): string {
  const match = specYaml.match(/^name:\s*(\S+)/m);
  return match ? match[1] : "study";
}
