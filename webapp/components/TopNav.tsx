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
          <Link href="#features">Features</Link>
          <Link href="#workspace">Workspace</Link>
          <Link href="#docs">Docs</Link>
          <Link href="#start" className="button primary" aria-label="Get started">
            Get started
          </Link>
        </div>
      </div>
    </nav>
  );
}
