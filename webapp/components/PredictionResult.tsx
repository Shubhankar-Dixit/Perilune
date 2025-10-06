import type { PredictResponse } from "../lib/api";

export function PredictionResult({ result }: { result: PredictResponse }) {
  const { label, probability, threshold, features, evidence } = result;
  const percent = (probability * 100).toFixed(1);
  const statusColor = label === "planet-candidate" ? "#6be3a3" : "#ff9180";

  return (
    <section className="card" aria-live="polite">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0, fontSize: "1.5rem" }}>Prediction</h2>
        <span
          style={{
            background: statusColor,
            color: "#05060b",
            padding: "0.35rem 0.75rem",
            borderRadius: "999px",
            fontWeight: 600,
            textTransform: "capitalize",
          }}
        >
          {label.replace("-", " ")}
        </span>
      </header>

      <div style={{ marginTop: "1rem", fontSize: "2rem", fontWeight: 600 }}>
        {percent}% confidence (threshold {Math.round(threshold * 100)}%)
      </div>

      <dl
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: "1rem",
          marginTop: "1.5rem",
        }}
      >
        <div>
          <dt style={{ opacity: 0.6 }}>Period (days)</dt>
          <dd style={{ margin: 0, fontSize: "1.1rem" }}>{features.period_days.toFixed(3)}</dd>
        </div>
        <div>
          <dt style={{ opacity: 0.6 }}>Duration (hours)</dt>
          <dd style={{ margin: 0, fontSize: "1.1rem" }}>{features.duration_hours.toFixed(3)}</dd>
        </div>
        <div>
          <dt style={{ opacity: 0.6 }}>Depth (ppm)</dt>
          <dd style={{ margin: 0, fontSize: "1.1rem" }}>{features.depth_ppm.toFixed(0)}</dd>
        </div>
        <div>
          <dt style={{ opacity: 0.6 }}>SNR</dt>
          <dd style={{ margin: 0, fontSize: "1.1rem" }}>{features.snr.toFixed(1)}</dd>
        </div>
      </dl>

      <section style={{ marginTop: "1.5rem" }}>
        <h3 style={{ marginTop: 0, fontSize: "1.1rem", opacity: 0.75 }}>Evidence</h3>
        <ul style={{ paddingLeft: "1.25rem", margin: 0 }}>
          {evidence.map((item) => (
            <li key={item} style={{ marginBottom: "0.35rem" }}>
              {item}
            </li>
          ))}
        </ul>
      </section>
    </section>
  );
}

