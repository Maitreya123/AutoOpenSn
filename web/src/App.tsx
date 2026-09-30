import { useEffect, useState } from "react";
import { api, pollJob, type Health, type Job, type Narrative, type SpecDraft, type TableRow } from "./api";
import Step from "./components/Step";
import ResultsTable from "./components/ResultsTable";
import Chat, { type Turn } from "./components/Chat";

// Four steps, in order, and nothing to configure.
//
// The engine behind this has runners, caches, fixture directories, retry
// budgets and a reviewer. None of that appears here. A person describing a
// physics study should not have to know what a fixture replay is, and the API
// picks the best available runner itself. The one thing that is never hidden is
// *which* engine produced the numbers, because a results table that does not
// say whether it came from OpenSn or from this package's own solver is a table
// nobody should quote.

const EXAMPLE = "compare GMRES tolerances 1e-4, 1e-6 and 1e-8 on the 1D transport problem";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  const [prompt, setPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [draft, setDraft] = useState<SpecDraft | null>(null);
  const [specError, setSpecError] = useState<string | null>(null);

  const [job, setJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [narrative, setNarrative] = useState<Narrative | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [thinking, setThinking] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const table: TableRow[] = job?.table ?? [];
  const haveResults = job?.status === "succeeded" && table.length > 0;

  async function generate() {
    setGenerating(true);
    setSpecError(null);
    // Everything downstream describes the previous question, so it goes.
    setDraft(null);
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
    if (!draft?.spec_yaml) return;
    setRunning(true);
    setRunError(null);
    setNarrative(null);
    setTurns([]);
    try {
      const started = await api.startRun(draft.spec_yaml);
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
      const result = await api.explain(table, draft?.spec_yaml ?? null);
      setNarrative(result);
      setTurns([
        {
          role: "assistant",
          content: result.text,
          warnings: result.warnings ?? result.unsupported,
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
        draft?.spec_yaml ?? null,
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

  function downloadSpec() {
    if (!draft?.spec_yaml) return;
    const blob = new Blob([draft.spec_yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${nameOf(draft.spec_yaml)}.yaml`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const percent = job?.total ? Math.round((job.completed / job.total) * 100) : 0;

  return (
    <div className="shell">
      <header className="masthead">
        <h1>AutoOpenSn</h1>
        <p>
          Describe a neutron transport study in plain language. It becomes a
          validated input spec, the spec is run, and the results are explained.
        </p>
        <div className="badges">
          <span className="badge">{health ? `${health.scenarios} scenarios` : "connecting…"}</span>
          <span className={`badge ${health?.llm_available ? "on" : "off"}`}>
            {health?.llm_available ? "model ready" : "no model"}
          </span>
          {health ? <span className="badge">OpenSn @ {health.pack_commit.slice(0, 7)}</span> : null}
        </div>
      </header>

      {/* 1 --------------------------------------------------------------- */}
      <Step number={1} title="What do you want to find out?" state={draft?.ok ? "done" : "active"}>
        <textarea
          rows={3}
          value={prompt}
          placeholder={EXAMPLE}
          onChange={(event) => setPrompt(event.target.value)}
          disabled={generating}
        />
        <div className="row">
          <button onClick={generate} disabled={generating || !prompt.trim()}>
            {generating ? <><span className="spinner" />Writing the spec…</> : "Generate spec"}
          </button>
          {!prompt && (
            <button className="quiet" onClick={() => setPrompt(EXAMPLE)} disabled={generating}>
              Use an example
            </button>
          )}
        </div>
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
        aside={draft?.ok ? `${draft.cases} run${draft.cases === 1 ? "" : "s"} · ${draft.summary}` : undefined}
      >
        {!draft?.ok ? (
          <p className="hint">Appears once a spec has been written.</p>
        ) : (
          <>
            <pre className="spec">{draft.spec_yaml}</pre>
            {draft.notes.length > 0 && (
              <div className="notice warn">{draft.notes.join(" ")}</div>
            )}
            <div className="row">
              <button onClick={run} disabled={running}>
                {running ? <><span className="spinner" />Running…</> : "Run this"}
              </button>
              <button className="quiet" onClick={downloadSpec}>
                Download spec
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
