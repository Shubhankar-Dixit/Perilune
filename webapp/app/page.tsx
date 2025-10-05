"use client";

import { useState, useTransition } from "react";
import { PredictionForm } from "../components/PredictionForm";
import { PredictionResult } from "../components/PredictionResult";
import { GraphPlaceholder } from "../components/GraphPlaceholder";
import { DataTablePlaceholder } from "../components/DataTablePlaceholder";
import { NotesPanel } from "../components/NotesPanel";
import { predictByObjectId, predictBySeries, type PredictResponse } from "../lib/api";

export default function HomePage() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  const handleCatalog = async (objectId: string, mission: string) => {
    startTransition(async () => {
      try {
        setError(null);
        const prediction = await predictByObjectId(objectId, mission);
        setResult(prediction);
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : "Prediction failed");
      }
    });
  };

  const handleUpload = async (times: number[], flux: number[], mission: string) => {
    startTransition(async () => {
      try {
        setError(null);
        const prediction = await predictBySeries(times, flux, mission);
        setResult(prediction);
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : "Prediction failed");
      }
    });
  };

  return (
    <main>
      <header className="hero">
        <h1>Perilune Explorer</h1>
        <p>
          Explore light curves, run predictions, and collect evidence. Spaces for graphs and notes below are ready
          to wire up to real data.
        </p>
      </header>

      <div className="dashboard-grid">
        <div className="col-left">
          <PredictionForm
            isLoading={isPending}
            onSubmitCatalog={handleCatalog}
            onSubmitUpload={handleUpload}
          />
          <NotesPanel />
          <DataTablePlaceholder title="Detections" />
        </div>

        <div className="col-right">
          {error ? (
            <section className="card" style={{ borderColor: "#ff7d7d" }}>
              <h2 style={{ marginTop: 0, color: "#ff7d7d" }}>Error</h2>
              <p>{error}</p>
            </section>
          ) : null}

          <GraphPlaceholder title="Light Curve" height={260} />
          <GraphPlaceholder title="BLS Periodogram" height={220} />
          {result ? (
            <PredictionResult result={result} />
          ) : (
            <section className="card" style={{ opacity: 0.8 }}>
              <h2 style={{ marginTop: 0 }}>Awaiting Input</h2>
              <p style={{ marginBottom: 0 }}>
                Run a prediction to see probabilities, features, and evidence appear here.
              </p>
            </section>
          )}
          <GraphPlaceholder title="Phase‑folded Plot" height={220} />
        </div>
      </div>
    </main>
  );
}

