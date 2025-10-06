import { describe, expect, it, vi } from "vitest";

vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ label: "planet-candidate", probability: 0.9, threshold: 0.5, features: { period_days: 10, duration_hours: 2, depth_ppm: 500, snr: 12 }, evidence: [] }) })));

import { predictByObjectId } from "../lib/api";

describe("predictByObjectId", () => {
  it("calls the API with the provided object id", async () => {
    await predictByObjectId("KOI-1", "kepler");
    expect(fetch).toHaveBeenCalledOnce();
    const [url, options] = (fetch as unknown as vi.Mock).mock.calls[0];
    expect(url).toContain("/api/predict");
    expect(options).toMatchObject({ method: "POST" });
  });
});

