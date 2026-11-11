import { lazy, Suspense, useEffect, useState } from "react";
import { api, type CaseFields, type FieldFile, type Spec } from "../api";

const Chart = lazy(() => import("./Chart"));

// Pictures of the fields a study's cases wrote: flux on the mesh, material
// regions, and for multigroup flux its energy spectrum. The server draws the
// pictures and works out the charts; this lays them out.

function groupLabel(array: string): string {
  const match = array.match(/^phi_g(\d+)_m(\d+)$/);
  if (!match) return array;
  const moment = Number(match[2]);
  return `Group ${Number(match[1])}${moment ? `, moment ${moment}` : ""}`;
}

function FieldView({ studyId, caseId, file }: { studyId: string; caseId: string; file: FieldFile }) {
  const [array, setArray] = useState<string | null>(null);
  const [chart, setChart] = useState<Spec | null>(null);
  const [broken, setBroken] = useState(false);

  // The spectrum, for multigroup flux, or the profile, for a CSV.
  useEffect(() => {
    if (file.kind !== "profile" && file.groups < 2) return;
    api.fieldChart(studyId, caseId, file.path).then((r) => setChart(r.chart)).catch(() => setChart(null));
  }, [studyId, caseId, file]);

  const name = file.path.split("/").pop();
  return (
    <div className="field">
      <div className="field-head">
        <span className="field-name" title={file.path}>{name}</span>
        {file.pieces > 1 && <span className="hint">{file.pieces} rank pieces, joined</span>}
        {file.kind === "vtk" && file.arrays.length > 1 && (
          <select
            value={array ?? ""}
            onChange={(event) => {
              setBroken(false);
              setArray(event.target.value || null);
            }}
            aria-label={`Field to draw from ${name}`}
          >
            <option value="">Default ({groupLabel(file.default_array ?? "")})</option>
            {file.arrays.map((a) => (
              <option key={a} value={a}>{groupLabel(a)}</option>
            ))}
          </select>
        )}
      </div>
      {file.kind === "vtk" && !broken && (
        <img
          className="field-image"
          src={api.fieldImage(studyId, caseId, file.path, array)}
          alt={`${array ?? file.default_array ?? "field"} from ${name}`}
          loading="lazy"
          onError={() => setBroken(true)}
        />
      )}
      {broken && <div className="notice bad">This field could not be drawn.</div>}
      {chart && (
        <Suspense fallback={null}>
          <Chart spec={chart} />
        </Suspense>
      )}
    </div>
  );
}

export default function Fields({ studyId }: { studyId: string }) {
  const [cases, setCases] = useState<CaseFields[] | null>(null);

  useEffect(() => {
    setCases(null);
    api.fields(studyId).then((r) => setCases(r.cases)).catch(() => setCases([]));
  }, [studyId]);

  if (!cases || cases.length === 0) return null;
  return (
    <div className="fields">
      <p className="examples-head">Fields the simulation wrote</p>
      {cases.map((c) => (
        <div key={c.case_id} className="field-case">
          {cases.length > 1 && <p className="field-case-label">{c.label}</p>}
          <div className="field-grid">
            {c.fields.map((f) => (
              <FieldView key={f.path} studyId={studyId} caseId={c.case_id} file={f} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
