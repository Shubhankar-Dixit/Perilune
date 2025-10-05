"use client";

export default function BackgroundOrbits() {
  return (
    <div aria-hidden className="bg-orbits">
      <svg width="100%" height="100%" viewBox="0 0 1600 900" preserveAspectRatio="none">
        <defs>
          <path id="orbitPath1" d="M 100 450 C 400 50, 1200 50, 1500 450 S 1200 850, 800 450 200 850 100 450" />
          <path id="orbitPath2" d="M 0 600 C 300 200, 900 200, 1600 600 S 900 1000, 300 700 0 800 0 600" />
        </defs>
        <g fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="1.2" strokeDasharray="6 10">
          <use href="#orbitPath1" className="dash-move slow-1" />
          <use href="#orbitPath2" className="dash-move slow-2" />
        </g>
      </svg>
    </div>
  );
}
