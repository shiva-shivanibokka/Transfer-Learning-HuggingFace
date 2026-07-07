// Client for the JSON inference API hosted on the Hugging Face Space.
// Override with NEXT_PUBLIC_API_URL for local backends.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "https://shiva-1993-transfer-learning-project.hf.space";

export type Prob = { label: string; prob: number };

export type VisionResult = {
  label: string;
  confidence: number;
  probabilities: Prob[];
  attention_png: string | null;
  latency_ms: number;
  device: string;
};

export type TextResult = {
  label: string;
  confidence: number;
  raw: Prob[];
  calibrated: Prob[];
  temperature: number;
};

export type ClipHit = { image_png: string; label: string; similarity: number };
export type ClipResult = { results: ClipHit[]; query: string };

export type VisionModel = {
  key: string; hub_id: string; family: string; year: number;
  params_m: number; accuracy: number; latency_ms: number; onnx_ms: number; has_attention: boolean;
};
export type TextModel = {
  key: string; hub_id: string; params_m: number; year: number; accuracy: number; temperature: number;
};
export type ModelsPayload = {
  vision: VisionModel[]; text: TextModel[];
  eurosat_classes: string[]; emotion_classes: string[];
};

export type ResultsPayload = {
  vision_strategy: Record<string, string | number>[];
  vision_data_efficiency: Record<string, string | number>[];
  text: Record<string, string | number>[];
  clip_prompts: { template: string; accuracy: number; ensemble?: boolean }[];
  findings: string[];
};

async function post<T>(path: string, body: unknown, timeoutMs = 120000): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`${res.status}: ${detail.slice(0, 200)}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

async function get<T>(path: string, timeoutMs = 30000): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

export const api = {
  models: () => get<ModelsPayload>("/api/models"),
  results: () => get<ResultsPayload>("/api/results"),
  health: () => get<{ status: string }>("/health", 8000),
  vision: (image: string, model: string) =>
    post<VisionResult>("/api/vision", { image, model }),
  text: (text: string, model: string) =>
    post<TextResult>("/api/text", { text, model }),
  clip: (query: string, k: number) =>
    post<ClipResult>("/api/clip-search", { query, k }),
};
