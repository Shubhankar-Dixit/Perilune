"use client";

import React from "react";

type Props = {
  title: string;
  height?: number;
  times?: number[] | null;
  flux?: number[] | null;
  features?: { period_days: number; duration_hours: number; depth_ppm: number } | null;
};

function fold(times: number[], period: number, epoch?: number) {
  const e = epoch ?? (times.length ? Math.min(...times) : 0);
  return times.map((t) => ((((t - e) % period) + period) % period) / period);
}

function syntheticPhase(samples = 300, periodDays = 10, durationHours = 2, depthPpm = 500) {
  const xs = Array.from({ length: samples }, (_, i) => i / (samples - 1));
  const durationFrac = Math.max(durationHours / 24 / periodDays, 0.01);
  const depth = depthPpm / 1e6;
  const ys = xs.map((x) => (x > 0.5 - durationFrac / 2 && x < 0.5 + durationFrac / 2 ? 1 - depth : 1));
  return { xs, ys };
}

export function PhaseFoldedGraph({ title, height = 220, times, flux, features }: Props) {
  let xs: number[] = [];
  let ys: number[] = [];
  const width = 800;

  const period = features?.period_days ?? 10;

  if (times && flux && times.length === flux.length && times.length > 1) {
    const phases = fold(times, period);
    const pairs = phases.map((p, i) => [p, flux[i]] as const).sort((a, b) => a[0] - b[0]);
    xs = pairs.map((p) => p[0]);
    ys = pairs.map((p) => p[1]);
  } else {
    const syn = syntheticPhase(320, period, features?.duration_hours ?? 2, features?.depth_ppm ?? 500);
    xs = syn.xs;
    ys = syn.ys;
  }

  const pad = 12;
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const ySpan = (maxY - minY) || 1;
  const points = xs
    .map((x, i) => {
      const px = pad + x * (width - pad * 2);
      const py = pad + (1 - (ys[i] - minY) / ySpan) * (height - pad * 2);
      return `${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");

  return (
    <section className="card">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <span className="pill" aria-hidden>Graph</span>
      </header>
      <div style={{ marginTop: "1rem" }}>
        <svg role="img" aria-label={title} width="100%" height={height} viewBox={`0 0 ${width} ${height}`}
          style={{ borderRadius: 12, border: "1px solid var(--border)", background: "rgba(255,255,255,0.01)" }}>
          <polyline points={points} fill="none" stroke="#f7b3b3" strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round" />
        </svg>
      </div>
    </section>
  );
}
