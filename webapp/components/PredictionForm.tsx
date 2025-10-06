"use client";

import { useState } from "react";
import { z } from "zod";

const missionOptions = [
  { value: "kepler", label: "Kepler" },
  { value: "k2", label: "K2" },
  { value: "tess", label: "TESS" },
];

const uploadSchema = z.object({
  times: z.string(),
  flux: z.string(),
  mission: z.enum(["kepler", "k2", "tess"]),
});

export type FormMode = "catalog" | "upload";

export type PredictionFormState = {
  mode: FormMode;
  objectId: string;
  mission: string;
  times: string;
  flux: string;
};

const initialState: PredictionFormState = {
  mode: "catalog",
  objectId: "",
  mission: "kepler",
  times: "",
  flux: "",
};

export function PredictionForm({
  isLoading,
  onSubmitCatalog,
  onSubmitUpload,
}: {
  isLoading: boolean;
  onSubmitCatalog: (objectId: string, mission: string) => Promise<void>;
  onSubmitUpload: (times: number[], flux: number[], mission: string) => Promise<void>;
}) {
  const [state, setState] = useState(initialState);
  const [error, setError] = useState<string | null>(null);

  const handleModeChange = (mode: FormMode) => {
    setState((prev) => ({ ...prev, mode }));
    setError(null);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (state.mode === "catalog") {
      if (!state.objectId.trim()) {
        setError("Enter a KOI/TOI/K2 identifier.");
        return;
      }
      await onSubmitCatalog(state.objectId.trim(), state.mission);
      return;
    }

    const parsed = uploadSchema.safeParse({
      times: state.times,
      flux: state.flux,
      mission: state.mission,
    });

    if (!parsed.success) {
      setError("Provide comma-separated numeric times and flux values.");
      return;
    }

    const times = parsed.data.times
      .split(",")
      .map((v) => Number.parseFloat(v.trim()))
      .filter((v) => Number.isFinite(v));
    const flux = parsed.data.flux
      .split(",")
      .map((v) => Number.parseFloat(v.trim()))
      .filter((v) => Number.isFinite(v));

    if (times.length !== flux.length || times.length < 5) {
      setError("Times and flux arrays must have the same length (>=5).");
      return;
    }

    await onSubmitUpload(times, flux, parsed.data.mission);
  };

  return (
    <form className="card" onSubmit={handleSubmit}>
      <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1rem" }}>
        <button
          type="button"
          className="button"
          onClick={() => handleModeChange("catalog")}
          disabled={isLoading || state.mode === "catalog"}
        >
          Catalog Lookup
        </button>
        <button
          type="button"
          className="button"
          onClick={() => handleModeChange("upload")}
          disabled={isLoading || state.mode === "upload"}
        >
          Upload Time Series
        </button>
      </div>

      {state.mode === "catalog" ? (
        <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "1rem" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            Object ID
            <input
              value={state.objectId}
              onChange={(e) => setState((prev) => ({ ...prev, objectId: e.target.value }))}
              placeholder="KOI-1234"
              disabled={isLoading}
              style={{ padding: "0.75rem", borderRadius: 12, border: "1px solid var(--border)" }}
            />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            Mission
            <select
              value={state.mission}
              onChange={(e) => setState((prev) => ({ ...prev, mission: e.target.value }))}
              disabled={isLoading}
              style={{ padding: "0.75rem", borderRadius: 12, border: "1px solid var(--border)" }}
            >
              {missionOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : (
        <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "1rem" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            Mission
            <select
              value={state.mission}
              onChange={(e) => setState((prev) => ({ ...prev, mission: e.target.value }))}
              disabled={isLoading}
              style={{ padding: "0.75rem", borderRadius: 12, border: "1px solid var(--border)" }}
            >
              {missionOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            Times (comma separated, days)
            <textarea
              value={state.times}
              onChange={(e) => setState((prev) => ({ ...prev, times: e.target.value }))}
              disabled={isLoading}
              rows={3}
              style={{ padding: "0.75rem", borderRadius: 12, border: "1px solid var(--border)", background: "transparent" }}
            />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            Flux (comma separated, normalized)
            <textarea
              value={state.flux}
              onChange={(e) => setState((prev) => ({ ...prev, flux: e.target.value }))}
              disabled={isLoading}
              rows={3}
              style={{ padding: "0.75rem", borderRadius: 12, border: "1px solid var(--border)", background: "transparent" }}
            />
          </label>
        </div>
      )}

      {error ? (
        <p style={{ color: "#ff7d7d", marginTop: "1rem" }}>{error}</p>
      ) : null}

      <button className="button primary" type="submit" disabled={isLoading} style={{ marginTop: "1.5rem" }}>
        {isLoading ? "Running..." : "Run Prediction"}
      </button>
    </form>
  );
}

