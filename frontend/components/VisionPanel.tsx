"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type VisionModel, type VisionResult } from "@/lib/api";
import { ErrorNote, Metric, Panel, ProbBars, SectionTitle, Spinner } from "@/components/ui";

const SAMPLES = ["river", "forest", "residential", "highway", "industrial", "annualcrop"];
const MAX_UPLOAD_BYTES = 2 * 1024 * 1024; // EuroSAT patches are tiny; reject large uploads.

function readAsDataURL(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("couldn't read the file"));
    reader.readAsDataURL(blob);
  });
}

export default function VisionPanel({ models }: { models: VisionModel[] }) {
  const [model, setModel] = useState(models.find((m) => m.key === "DINOv2-Base")?.key ?? models[0]?.key);
  // For samples, `imgUrl` is the static /samples display source (distinct from
  // the base64 we send). For uploads it stays null and the preview derives from
  // `imgB64`, so the large data URL is never held in state twice.
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [imgB64, setImgB64] = useState<string | null>(null);
  const [res, setRes] = useState<VisionResult | null>(null);
  const [resModel, setResModel] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const reqIdRef = useRef(0);
  const ctrlRef = useRef<AbortController | null>(null);

  const preview = imgUrl ?? imgB64;
  const selMeta = models.find((m) => m.key === model);
  const resMeta = models.find((m) => m.key === resModel);
  // Lock the output panel's meta to the model that produced the current result;
  // fall back to the selected model before any result exists.
  const headMeta = res ? resMeta : selMeta;

  const loadFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) { setErr("Please upload an image file."); return; }
    if (file.size > MAX_UPLOAD_BYTES) { setErr("Image too large — please use a patch under 2 MB."); return; }
    readAsDataURL(file)
      .then((url) => { setImgUrl(null); setImgB64(url); setRes(null); setErr(null); })
      .catch(() => setErr("Couldn't read the selected file."));
  }, []);

  async function loadSample(name: string) {
    const url = `/samples/${name}.png`;
    setErr(null); setRes(null);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status}`);
      const b64 = await readAsDataURL(await res.blob());
      setImgB64(b64);
      setImgUrl(url); // set the preview only after base64 conversion succeeds
    } catch {
      setErr(`Couldn't load the "${name}" sample.`);
    }
  }

  // Abort any in-flight request when the panel unmounts (e.g. tab switch).
  useEffect(() => () => ctrlRef.current?.abort(), []);

  async function classify() {
    if (!imgB64 || !model) return;
    const reqModel = model;               // snapshot the model this request used
    const reqId = ++reqIdRef.current;     // monotonic guard against stale responses
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setLoading(true); setErr(null); setRes(null);
    try {
      const r = await api.vision(imgB64, reqModel, ctrl.signal);
      if (reqId !== reqIdRef.current) return; // superseded — discard
      setRes(r); setResModel(reqModel);
    } catch (e) {
      if (reqId !== reqIdRef.current || ctrl.signal.aborted) return;
      setErr(e instanceof Error ? e.message : "request failed");
    } finally {
      if (reqId === reqIdRef.current) setLoading(false);
    }
  }

  return (
    <div className="grid lg:grid-cols-2 gap-5">
      {/* Input */}
      <Panel>
        <SectionTitle kicker="EuroSAT · 10 classes" title="Land-use classifier"
          sub="Upload a satellite patch (or pick a sample), choose a backbone, classify."
          help="Classifies a Sentinel-2 satellite patch into one of 10 EuroSAT land-use classes. Pick a backbone (CNNs vs. transformers) to compare; each is fine-tuned and served from the Hugging Face Hub." />

        <div className="flex flex-wrap gap-2 mb-4">
          {models.map((m) => (
            <button key={m.key} onClick={() => { setModel(m.key); setRes(null); }}
              disabled={loading}
              className="chip" data-active={m.key === model} aria-pressed={m.key === model}
              style={m.key === model ? { borderColor: "var(--accent)", color: "var(--fg)" } : {}}>
              {m.key}
            </button>
          ))}
        </div>

        <div
          role="button" tabIndex={0}
          aria-label="Upload an image: drop a file or activate to browse"
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileRef.current?.click(); } }}
          onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) loadFile(f); }}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
          className="rounded-xl border border-dashed flex items-center justify-center cursor-pointer overflow-hidden"
          style={{ borderColor: "var(--border-hi)", background: "var(--panel-2)", height: 240 }}>
          {preview ? (
            <img src={preview} alt="input" className="h-full w-full object-cover" />
          ) : (
            <div className="text-center text-sm" style={{ color: "var(--faint)" }}>
              <div className="text-2xl mb-1">⊕</div>drop image or click to upload
            </div>
          )}
        </div>
        <input ref={fileRef} type="file" accept="image/*" className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(f); }} />

        <div className="mt-3">
          <div className="mono text-[10px] uppercase tracking-wider mb-1.5" style={{ color: "var(--faint)" }}>samples</div>
          <div className="grid grid-cols-6 gap-1.5">
            {SAMPLES.map((s) => (
              <button key={s} onClick={() => loadSample(s)}
                className="aspect-square rounded-md overflow-hidden border hover:opacity-80 transition"
                style={{ borderColor: "var(--border)" }} title={s}>
                {/* plain img: samples are tiny + static; avoids the next/image optimizer round-trip */}
                <img src={`/samples/${s}.png`} alt={s} className="w-full h-full object-cover" />
              </button>
            ))}
          </div>
        </div>

        <button onClick={classify} disabled={!imgB64 || loading}
          className="btn-accent w-full mt-4 py-2.5 text-sm">
          {loading ? "Running inference…" : "Classify"}
        </button>
      </Panel>

      {/* Output */}
      <Panel>
        <SectionTitle kicker={headMeta ? `${headMeta.family} · ${headMeta.year} · ${headMeta.params_m}M params` : ""} title="Prediction"
          help="Predicted class and confidence, the full 10-class probability distribution, single-image inference latency, and — for ViT/DINOv2 — an attention-rollout heatmap showing which image regions the model focused on." />
        {!res && !loading && !err && (
          <div className="text-sm py-16 text-center" style={{ color: "var(--faint)" }}>
            Results appear here. First call after the Space sleeps downloads the model — give it a few seconds.
          </div>
        )}
        {loading && <div className="py-16 flex justify-center"><Spinner label="loading model from the Hub + inferring…" /></div>}
        {err && <ErrorNote msg={err} />}
        {res && (
          <div className="space-y-4">
            <div className="flex items-end justify-between">
              <div>
                <div className="mono text-[10px] uppercase tracking-wider" style={{ color: "var(--faint)" }}>predicted</div>
                <div className="text-2xl font-semibold" style={{ color: "var(--accent)" }}>{res.label}</div>
              </div>
              <div className="mono text-3xl" style={{ color: "var(--fg)" }}>{(res.confidence * 100).toFixed(1)}<span className="text-lg" style={{ color: "var(--muted)" }}>%</span></div>
            </div>
            <ProbBars items={res.probabilities} topN={6} />
            <div className="grid grid-cols-3 gap-2">
              <Metric label="latency" value={res.latency_ms.toFixed(1)} unit="ms" />
              <Metric label="device" value={res.device.toUpperCase()} />
              <Metric label="test acc" value={resMeta ? resMeta.accuracy.toFixed(1) : "—"} unit="%" accent="var(--accent2)" />
            </div>
            {res.attention_png && (
              <div>
                <div className="mono text-[10px] uppercase tracking-wider mb-1.5" style={{ color: "var(--faint)" }}>attention rollout</div>
                <img src={res.attention_png} alt="attention" className="rounded-lg w-full max-w-[240px]" />
              </div>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}
