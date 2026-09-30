import type { ReactNode } from "react";

// A step is either the one you are on, one you have finished, or one that is
// not yet reachable. Showing the unreachable ones greyed rather than hiding
// them is the point of a guided flow: you can see where this is going.
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
      {children}
    </section>
  );
}
