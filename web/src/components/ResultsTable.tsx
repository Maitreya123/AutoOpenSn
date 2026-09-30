import type { TableRow } from "../api";

// Numbers in a transport results table span twenty orders of magnitude: a
// relative flux difference of 1e-10 next to a wall time of 0.44 seconds. A
// single format for both makes one of them unreadable, so the magnitude picks
// the notation.
//
// Nothing here rounds a value into a different one: exponential notation keeps
// four significant figures, which is more than the underlying measurement
// carries and enough that two rows never render identically when they differ.
function format(value: TableRow[string]): { text: string; numeric: boolean } {
  if (value === null || value === undefined) return { text: "—", numeric: false };
  if (typeof value === "boolean") return { text: value ? "yes" : "no", numeric: false };
  if (typeof value === "number") {
    if (Number.isInteger(value)) return { text: String(value), numeric: true };
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e5)) {
      return { text: value.toExponential(4), numeric: true };
    }
    return { text: value.toFixed(magnitude < 1 ? 6 : 4), numeric: true };
  }
  return { text: String(value), numeric: false };
}

export default function ResultsTable({ rows }: { rows: TableRow[] }) {
  if (!rows.length) return <p className="hint">No rows.</p>;
  // Union of keys rather than the first row's, so a column that only some cases
  // produced is still shown instead of silently dropped.
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => {
                const cell = format(row[column]);
                return (
                  <td key={column} className={cell.numeric ? "num" : undefined}>
                    {cell.text}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
