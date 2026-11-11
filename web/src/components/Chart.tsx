import { useEffect, useRef, useState } from "react";
import vegaEmbed, { type Result, type VisualizationSpec } from "vega-embed";

// Draws one Vega-Lite specification. The specifications come from the server,
// worked out from the study itself (see autoopensn/viz/charts.py); this only
// renders them, so the page and a saved report always show the same chart.

function currentTheme(): "light" | "dark" {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

export default function Chart({ spec }: { spec: Record<string, unknown> }) {
  const container = useRef<HTMLDivElement>(null);
  const [theme, setTheme] = useState(currentTheme);
  const [error, setError] = useState<string | null>(null);

  // Follow the light/dark toggle, which lives in the masthead: redraw when the
  // page's theme attribute changes rather than threading state down to here.
  useEffect(() => {
    const watcher = new MutationObserver(() => setTheme(currentTheme()));
    watcher.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => watcher.disconnect();
  }, []);

  useEffect(() => {
    if (!container.current) return;
    let view: Result | null = null;
    let cancelled = false;
    setError(null);
    vegaEmbed(
      container.current,
      { ...spec, background: "transparent" } as VisualizationSpec,
      {
        actions: false,
        renderer: "svg",
        ...(theme === "dark" ? { theme: "dark" as const } : {}),
      },
    )
      .then((result) => {
        if (cancelled) result.finalize();
        else view = result;
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
    return () => {
      cancelled = true;
      view?.finalize();
    };
  }, [spec, theme]);

  return (
    <div className="chart">
      <div ref={container} />
      {error && <div className="notice bad">This chart could not be drawn: {error}</div>}
    </div>
  );
}
