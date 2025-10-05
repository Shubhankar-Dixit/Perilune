export function DataTablePlaceholder({ title = "Data Table" }: { title?: string }) {
  return (
    <section className="card">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <span className="pill" aria-hidden>Table</span>
      </header>
      <div className="placeholder" style={{ height: 220 }} aria-label={`${title} placeholder`} />
    </section>
  );
}
