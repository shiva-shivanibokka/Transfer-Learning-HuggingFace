"use client";

import { useCallback, useRef, useState } from "react";
import { api, type VisionModel, type VisionResult } from "@/lib/api";
import { ErrorNote, Metric, Panel, ProbBars, SectionTitle, Spinner } from "@/components/ui";

const SAMPLES = ["river", "forest", "residential", "highway", "industrial", "annualcrop"];

export default function VisionPanel({ models }: { models: VisionModel[] }) {
  const [model, setModel] = useState(models.find((m) => m.key === "DINOv2-Base")?.key ?? models[0]?.key);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [imgB64, setImgB64] = useState<string | null>(null);
  const [res, setRes] = useState<VisionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const meta = models.find((m) => m.key === model);

  const loadFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = reader.result as string;
      setImgUrl(url);
      setImgB64(url);
      setRes(null);
      setErr(null);
    };
    reader.readAsDataURL(file);
  }, []);

  async function loadSample(name: string) {
    const url = `/samples/${name}.png`;
    setImgUrl(url); setRes(null); setErr(null);
    const blob = await fetch(url).then((r) => r.blob());
    const reader = new FileReader();
    reader.onload = () => setImgB64(reader.result as string);
    reader.readAsDataURL(blob);
  }

  async function classify() {
    if (!imgB64 || !model) return;
    setLoading(true); setErr(null); setRes(null);
    try {
      setRes(await api.vision(imgB64, model));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "request failed");
    } finally {
      setLoading(false);
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
              className="chip" data-active={m.key === model}
              style={m.key === model ? { borderColor: "var(--accent)", color: "var(--fg)" } : {}}>
              {m.key}
            </button>
          ))}
        </div>

        <div
          onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) loadFile(f); }}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
          className="rounded-xl border border-dashed flex items-center justify-center cursor-pointer overflow-hidden"
          style={{ borderColor: "var(--border-hi)", background: "var(--panel-2)", height: 240 }}>
          {imgUrl ? (
            <img src={imgUrl} alt="input" className="h-full w-full object-cover" />
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
        <SectionTitle kicker={meta ? `${meta.family} · ${meta.year} · ${meta.params_m}M params` : ""} title="Prediction"
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
              <Metric label="test acc" value={meta ? meta.accuracy.toFixed(1) : "—"} unit="%" accent="var(--accent2)" />
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
