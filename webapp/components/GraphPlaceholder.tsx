export function GraphPlaceholder({ title, height = 260 }: { title: string; height?: number }) {
  return (
    <section className="card">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <span className="pill" aria-hidden>Graph</span>
      </header>
      <div className="placeholder" style={{ height }} aria-label={`${title} placeholder`} />
    </section>
  );
}
