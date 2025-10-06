"use client";

import Link from "next/link";

export default function TopNav() {
  return (
    <nav className="topnav" aria-label="Primary">
      <div className="container" style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <Link href="/" className="brand" aria-label="Perilune Home">
          <span className="brand-mark">◷</span>
          <span className="brand-name">Perilune</span>
        </Link>
        <div style={{ flex: 1 }} />
        <div className="topnav-links">
          <Link href="/about">About</Link>
          <Link href="/how-it-works">How it works</Link>
          <Link href="https://github.com" target="_blank" rel="noreferrer">Docs</Link>
          <Link href="/" className="button primary" aria-label="Open app">Open app</Link>
        </div>
      </div>
    </nav>
  );
}
