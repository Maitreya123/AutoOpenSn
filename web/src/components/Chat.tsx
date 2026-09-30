import { useState } from "react";

export type Turn = { role: "user" | "assistant"; content: string; warnings?: string[] };

// Paragraphs only. The narration stage writes prose about a table, and rendering
// it as Markdown would mean interpreting model output as markup — which is how a
// stray asterisk in a parameter name turns half an answer italic.
function Prose({ text }: { text: string }) {
  return (
    <>
      {text
        .split(/\n\s*\n/)
        .filter((block) => block.trim())
        .map((block, index) => (
          <p key={index}>{block.trim()}</p>
        ))}
    </>
  );
}

export default function Chat({
  turns,
  busy,
  onAsk,
}: {
  turns: Turn[];
  busy: boolean;
  onAsk: (question: string) => void;
}) {
  const [draft, setDraft] = useState("");

  function send() {
    const question = draft.trim();
    if (!question || busy) return;
    setDraft("");
    onAsk(question);
  }

  return (
    <>
      <div className="chat">
        {turns.map((turn, index) => (
          <div key={index} className={`turn ${turn.role === "user" ? "reader" : "model"}`}>
            <Prose text={turn.content} />
            {turn.warnings?.length ? (
              <div className="notice warn">{turn.warnings.join(" ")}</div>
            ) : null}
          </div>
        ))}
        {busy ? (
          <div className="turn model">
            <span className="spinner" />
            Reading the table…
          </div>
        ) : null}
      </div>

      <div className="row">
        <input
          type="text"
          value={draft}
          placeholder="Ask about these results…"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") send();
          }}
          disabled={busy}
          style={{ flex: 1, minWidth: 220 }}
        />
        <button onClick={send} disabled={busy || !draft.trim()}>
          Ask
        </button>
      </div>
      <p className="hint" style={{ marginTop: 8 }}>
        Answers come from the table above and nothing else. Any figure that is
        not in it is flagged.
      </p>
    </>
  );
}
