import { useEffect, useState } from "react";

type Theme = "dark" | "light";
const STORAGE_KEY = "autoopensn.theme";

function stored(): Theme {
  // Wrapped because site data can be blocked outright, and a page that throws
  // here would fail to render at all over a preference.
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* no stored preference available */
  }
  return "dark";
}

// A slab, five regions, a beam entering from the left: the Reed problem, which
// is the scenario nearly every study here starts from. Drawn rather than
// imported so it inherits the theme's own colours.
function Mark() {
  return (
    <svg className="mark" width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">
      <rect
        x="6.5"
        y="10.5"
        width="27"
        height="19"
        rx="2.5"
        fill="none"
        stroke="var(--accent-bright)"
        strokeWidth="1.4"
        opacity="0.85"
      />
      {[13.5, 19, 24.5].map((x) => (
        <line
          key={x}
          x1={x}
          y1="10.5"
          x2={x}
          y2="29.5"
          stroke="var(--accent-bright)"
          strokeWidth="1"
          opacity="0.35"
        />
      ))}
      <path
        d="M0.5 20 H6"
        stroke="var(--accent-bright)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      <path
        d="M34 20 H39.5"
        stroke="var(--accent-bright)"
        strokeWidth="1.4"
        strokeLinecap="round"
        opacity="0.45"
      />
      <circle cx="10" cy="20" r="1.9" fill="var(--accent-bright)" />
    </svg>
  );
}

export default function Masthead() {
  const [theme, setTheme] = useState<Theme>(stored);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* the page still works; the choice just will not persist */
    }
  }, [theme]);

  return (
    <header className="masthead">
      <button
        type="button"
        className="theme-toggle"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        title={theme === "dark" ? "Light theme" : "Dark theme"}
      >
        {theme === "dark" ? (
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <circle cx="8" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.3" />
            {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
              <line
                key={a}
                x1="8"
                y1="1.4"
                x2="8"
                y2="3"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
                transform={`rotate(${a} 8 8)`}
              />
            ))}
          </svg>
        ) : (
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path
              d="M13.2 10.1A5.6 5.6 0 0 1 6 2.8a5.6 5.6 0 1 0 7.2 7.3Z"
              stroke="currentColor"
              strokeWidth="1.3"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </button>

      <Mark />
      <h1>AutoOpenSn</h1>
      <p>Describe a transport study in plain language. It becomes a validated spec, is run, and explained.</p>
    </header>
  );
}
