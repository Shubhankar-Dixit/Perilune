"use client";

import { useState } from "react";

export function NotesPanel() {
  const [text, setText] = useState("");
  return (
    <section className="card">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>Notes</h3>
        <span className="pill" aria-hidden>Discuss</span>
      </header>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Write observations, hypotheses, and follow‑ups…"
        rows={8}
        style={{ width: "100%", marginTop: "1rem", resize: "vertical", padding: "0.75rem", borderRadius: 8, border: "1px solid rgba(255,255,255,0.15)", background: "transparent", color: "inherit" }}
      />
    </section>
  );
}
