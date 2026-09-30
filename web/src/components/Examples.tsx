// Studies that actually produce a table on a machine with no OpenSn.
//
// Every one of these stays inside the two scenarios the built-in solver
// implements, `reed_1d` and `first_1d_fixed_source`, and inside the ranges
// their sidecars declare. That constraint is the whole point: an example that
// generates a valid spec and then reports every case as unavailable teaches a
// new user that the tool is broken.
//
// The label is what a person is trying to find out; the prompt is how they
// would say it. Both are shown, because the prompt is the thing being
// demonstrated — the text box is not a menu, and hiding the sentence would
// suggest it is.
export const EXAMPLES: { label: string; prompt: string }[] = [
  {
    label: "Convergence tolerance",
    prompt:
      "compare GMRES tolerances 1e-4, 1e-6 and 1e-8 on the 1D transport problem",
  },
  {
    label: "Which inner solver is fastest",
    prompt:
      "run the Reed problem with each inner solver — Richardson, GMRES and BiCGStab — and tell me which converges in the fewest sweeps",
  },
  {
    label: "GMRES restart interval",
    prompt:
      "on the Reed problem, compare GMRES restart intervals of 3, 5, 10 and 30 against a 1e-10 reference",
  },
  {
    label: "Angular refinement",
    prompt:
      "vary the number of polar angles over 16, 32, 64 and 128 on the Reed problem and show how the flux changes",
  },
  {
    label: "Mesh refinement",
    prompt:
      "refine the Reed mesh — 100, 200, 400 and 800 cells — and show the effect on the flux and the runtime",
  },
  {
    label: "Scattering ratio",
    prompt:
      "on the first 1D fixed-source problem, sweep the scattering ratio over 0.2, 0.5, 0.9 and 0.99 and tell me how the iteration count responds",
  },
];

export default function Examples({
  onPick,
  disabled,
}: {
  onPick: (prompt: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="examples">
      <p className="examples-head">Or start from one of these</p>
      <div className="example-grid">
        {EXAMPLES.map((example) => (
          <button
            key={example.label}
            type="button"
            className="example"
            onClick={() => onPick(example.prompt)}
            disabled={disabled}
          >
            <span className="example-label">{example.label}</span>
            <span className="example-prompt">{example.prompt}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
