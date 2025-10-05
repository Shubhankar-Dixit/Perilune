"use client";

import React from "react";

type Props = {
  title: string;
  height?: number;
  features?: { period_days: number } | null;
};

function makePeriodogram(width: number, samples = 500, bestPeriod = 10) {
  const minP = 0.5;
  const maxP = 40;
  const periods: number[] = Array.from({ length: samples }, (_, i) => minP + ((maxP - minP) * i) / (samples - 1));
  const sigma = Math.max(bestPeriod * 0.15, 0.8);
  const power: number[] = periods.map((p) => Math.exp(-0.5 * Math.pow((p - bestPeriod) / sigma, 2)));
  return { periods, power, domain: [minP, maxP] as const };
}

function toPolyline(periods: number[], power: number[], width: number, height: number, domain: readonly [number, number]) {
  const [minP, maxP] = domain;
  const pad = 12;
  return periods
    .map((p, i) => {
      const px = pad + ((p - minP) / (maxP - minP)) * (width - pad * 2);
      const py = pad + (1 - power[i]) * (height - pad * 2);
      return `${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

export function PeriodogramGraph({ title, height = 220, features }: Props) {
  const best = features?.period_days ?? 10;
  const width = 800;
  const data = makePeriodogram(width, 500, best);
  const points = toPolyline(data.periods, data.power, width, height, data.domain);

  return (
    <section className="card">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <span className="pill" aria-hidden>Graph</span>
      </header>
      <div style={{ marginTop: "1rem" }}>
        <svg role="img" aria-label={title} width="100%" height={height} viewBox={`0 0 ${width} ${height}`}
          style={{ borderRadius: 12, border: "1px solid var(--border)", background: "rgba(255,255,255,0.01)" }}>
          <polyline points={points} fill="none" stroke="#9bd3ff" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
          {/* Marker for best period */}
          <line x1={(best - 0.5) / (40 - 0.5) * (width - 24) + 12} x2={(best - 0.5) / (40 - 0.5) * (width - 24) + 12}
                y1={12} y2={height - 12} stroke="#ffffff33" strokeDasharray="4 4" />
        </svg>
      </div>
    </section>
  );
}
