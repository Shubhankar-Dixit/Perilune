export default function HowItWorksPage() {
  return (
    <main>
      <section className="hero">
        <h1>How it works</h1>
        <p>Upload a light curve or reference an object ID. We run BLS to extract features, then score with a model.</p>
      </section>
      <section className="card">
        <h2 style={{ marginTop: 0 }}>Pipeline</h2>
        <ol>
          <li>Fetch or accept light curve (time, flux).</li>
          <li>Run BLS search to estimate period, duration, depth, SNR.</li>
          <li>Score features; return probabilities with evidence.</li>
        </ol>
      </section>
    </main>
  );
}
