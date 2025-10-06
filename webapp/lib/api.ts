export type PredictResponse = {
  label: string;
  probability: number;
  threshold: number;
  features: {
    period_days: number;
    duration_hours: number;
    depth_ppm: number;
    snr: number;
  };
  evidence: string[];
};

const API_BASE = process.env.NEXT_PUBLIC_PERILUNE_API ?? "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function predictByObjectId(objectId: string, mission?: string) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ objectId, mission, dryRun: false }),
  });
  return handleResponse<PredictResponse>(res);
}

export async function predictBySeries(
  times: number[],
  flux: number[],
  mission: string,
) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ times, flux, mission, dryRun: false }),
  });
  return handleResponse<PredictResponse>(res);
}

export type SearchTransitsResponse = {
  result: {
    best_period_days: number;
    best_duration_hours: number;
    depth_ppm: number;
    snr: number;
    [key: string]: unknown;
  };
};

export async function searchTransits(times: number[], flux: number[]) {
  const res = await fetch(`${API_BASE}/api/search-transits`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ times, flux, dryRun: false }),
  });
  return handleResponse<SearchTransitsResponse>(res);
}

