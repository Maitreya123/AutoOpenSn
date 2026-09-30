import type { ReactNode } from "react";

// A step is either the one you are on, one you have finished, or one that is
// not yet reachable.
//
// A step that cannot be reached yet renders as a single line: its number and
// its title, and nothing else. It stays visible, because seeing where the flow
// leads is worth one line — but only one. Three full-height cards each saying
// "nothing here yet" is the same single fact told three times, and it costs a
// third of the page to say it.
export default function Step({
  number,
  title,
  aside,
  state,
  children,
}: {
  number: number;
  title: string;
  aside?: ReactNode;
  state: "waiting" | "active" | "done";
  children: ReactNode;
}) {
  return (
    <section className={`step ${state}`}>
      <div className="step-head">
        <span className="step-number">{state === "done" ? "✓" : number}</span>
        <span className="step-title">{title}</span>
        {aside ? <span className="step-sub">{aside}</span> : null}
      </div>
      {state === "waiting" ? null : children}
    </section>
  );
}
