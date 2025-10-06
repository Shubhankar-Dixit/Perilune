"use client";

import React from "react";

type Props = {
  title: string;
  height?: number;
  times?: number[] | null;
  flux?: number[] | null;
  features?: { period_days: number; duration_hours: number; depth_ppm: number } | null;
};

function makeSynthetic(timesCount = 400, periodDays = 10, durationHours = 2, depthPpm = 500) {
  const span = Math.max(periodDays * 3, 10);
  const times: number[] = Array.from({ length: timesCount }, (_, i) => (i / (timesCount - 1)) * span);
  const depth = depthPpm / 1e6;
  const durationDays = durationHours / 24;
  const flux: number[] = times.map((t) => {
    const phase = (t % periodDays) / periodDays;
    const inTransit = phase > 0.5 - durationDays / (2 * periodDays) && phase < 0.5 + durationDays / (2 * periodDays);
    const base = 1 - (inTransit ? depth : 0);
    return base;
  });
  return { times, flux };
}

function toPolylinePoints(xs: number[], ys: number[], width: number, height: number) {
  if (xs.length === 0 || ys.length === 0) return "";
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  const pad = 12;
  return xs
    .map((x, i) => {
      const y = ys[i];
      const px = pad + ((x - minX) / xSpan) * (width - pad * 2);
      const py = pad + (1 - (y - minY) / ySpan) * (height - pad * 2);
      return `${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

export function LightCurveGraph({ title, height = 260, times, flux, features }: Props) {
  let xs = times ?? [];
  let ys = flux ?? [];
  if (xs.length === 0 || ys.length === 0) {
    const p = features?.period_days ?? 10;
    const d = features?.duration_hours ?? 2;
    const dp = features?.depth_ppm ?? 500;
    const syn = makeSynthetic(400, p, d, dp);
    xs = syn.times;
    ys = syn.flux;
  }

  const width = 800;
  const points = toPolylinePoints(xs, ys, width, height);

  return (
    <section className="card">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <span className="pill" aria-hidden>
          Graph
        </span>
      </header>
      <div style={{ marginTop: "1rem" }}>
        <svg role="img" aria-label={title} width="100%" height={height} viewBox={`0 0 ${width} ${height}`}
          style={{ borderRadius: 12, border: "1px solid var(--border)", background: "rgba(255,255,255,0.01)" }}>
          <defs>
            <linearGradient id="lcFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(241,217,138,0.18)" />
              <stop offset="100%" stopColor="rgba(241,217,138,0.00)" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width={width} height={height} fill="url(#lcFill)" />
          <polyline points={points} fill="none" stroke="#f1d98a" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
        </svg>
      </div>
    </section>
  );
}
