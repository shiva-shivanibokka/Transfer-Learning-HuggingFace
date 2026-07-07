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

// Cold free HF Spaces can take a while to wake; the initial GET probes need a
// generous timeout so the first load doesn't falsely report a "waking" error.
const BOOTSTRAP_TIMEOUT = 90000;

// Combine an optional caller-supplied AbortSignal with an internal timeout so a
// request is aborted either on timeout or when the caller (e.g. an unmounting
// panel / superseding request) cancels it.
function linkSignal(timeoutMs: number, external?: AbortSignal) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  const onAbort = () => ctrl.abort();
  if (external) {
    if (external.aborted) ctrl.abort();
    else external.addEventListener("abort", onAbort, { once: true });
  }
  const cleanup = () => {
    clearTimeout(t);
    external?.removeEventListener("abort", onAbort);
  };
  return { signal: ctrl.signal, cleanup };
}

async function post<T>(
  path: string,
  body: unknown,
  opts: { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<T> {
  const { timeoutMs = 120000, signal: external } = opts;
  const { signal, cleanup } = linkSignal(timeoutMs, external);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`${res.status}: ${detail.slice(0, 200)}`);
    }
    return (await res.json()) as T;
  } finally {
    cleanup();
  }
}

async function get<T>(path: string, timeoutMs = 30000): Promise<T> {
  const { signal, cleanup } = linkSignal(timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, { signal });
    if (!res.ok) throw new Error(`${res.status}`);
    return (await res.json()) as T;
  } finally {
    cleanup();
  }
}

export const api = {
  // Bootstrap calls tolerate a cold Space waking from sleep.
  models: () => get<ModelsPayload>("/api/models", BOOTSTRAP_TIMEOUT),
  results: () => get<ResultsPayload>("/api/results", BOOTSTRAP_TIMEOUT),
  // Lightweight cold-start probe used to wake the Space before the heavier calls.
  health: () => get<{ status: string }>("/health", BOOTSTRAP_TIMEOUT),
  vision: (image: string, model: string, signal?: AbortSignal) =>
    post<VisionResult>("/api/vision", { image, model }, { signal }),
  text: (text: string, model: string, signal?: AbortSignal) =>
    post<TextResult>("/api/text", { text, model }, { signal }),
  clip: (query: string, k: number, signal?: AbortSignal) =>
    post<ClipResult>("/api/clip-search", { query, k }, { signal }),
};
