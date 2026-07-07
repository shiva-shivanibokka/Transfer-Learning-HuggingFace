"use client";

import { useEffect, useState } from "react";
import { api, type ModelsPayload, type ResultsPayload } from "@/lib/api";
import VisionPanel from "@/components/VisionPanel";
import TextPanel from "@/components/TextPanel";
import ClipPanel from "@/components/ClipPanel";
import ResultsPanel from "@/components/ResultsPanel";
import AboutPanel from "@/components/AboutPanel";
import { Spinner } from "@/components/ui";

const TABS = ["Vision", "Text", "CLIP Search", "Results", "About"] as const;
type Tab = (typeof TABS)[number];

// Plain-language, one-line intro shown as a banner when each tab is selected.
const TAB_DESC: Record<Tab, string> = {
  Vision: "Give it a satellite photo and four different AI models each guess what kind of land it shows — forest, city, river, farmland, and so on. Switch models to see how they compare.",
  Text: "Type any sentence and the model reads the emotion behind it (joy, fear, anger…), showing how confident it is — and how that confidence changes after a calibration step.",
  "CLIP Search": "Describe a scene in plain English and it finds the satellite images that best match your words — no labels or training needed, just meaning.",
  Results: "The evidence behind the demos: how accurate each model is, how they cope with very little training data, and how trustworthy their confidence scores are.",
  About: "",
};

const HF_SPACE = "https://huggingface.co/spaces/shiva-1993/transfer-learning-project";
const HF_MODELS = "https://huggingface.co/shiva-1993";
const GITHUB = "https://github.com/shiva-shivanibokka/Transfer-Learning-HuggingFace";

export default function Home() {
  const [tab, setTab] = useState<Tab>("Vision");
  const [models, setModels] = useState<ModelsPayload | null>(null);
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [status, setStatus] = useState<"connecting" | "live" | "error">("connecting");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        // Wake the Space with a cheap probe first; ignore its outcome so a
        // missing/failed /health endpoint never blocks the real bootstrap.
        await api.health().catch(() => {});
        if (!alive) return;
        const [m, r] = await Promise.all([api.models(), api.results()]);
        if (!alive) return;
        setModels(m); setResults(r); setStatus("live");
      } catch {
        if (alive) setStatus("error");
      }
    })();
    return () => { alive = false; };
  }, []);

  return (
    <div className="flex-1 flex flex-col max-w-6xl w-full mx-auto px-4 sm:px-6">
      <header className="pt-8 pb-5">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-2">
              <span className="mono text-[11px] tracking-widest uppercase" style={{ color: "var(--accent)" }}>Transfer-Learning Lab</span>
              <StatusBadge status={status} />
            </div>
            <h1 className="text-2xl sm:text-3xl font-semibold mt-1.5 leading-tight">
              Vision · Text · CLIP <span style={{ color: "var(--muted)" }}>— transfer learning on satellite & emotion data</span>
            </h1>
          </div>
          <nav className="flex gap-2 mono text-xs">
            <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="chip hover:opacity-80">GitHub</a>
            <a href={HF_MODELS} target="_blank" rel="noopener noreferrer" className="chip hover:opacity-80">🤗 Models</a>
            <a href={HF_SPACE} target="_blank" rel="noopener noreferrer" className="chip hover:opacity-80">Space</a>
          </nav>
        </div>
        <p className="text-sm mt-3" style={{ color: "var(--muted)" }}>
          Interactive demos of 4 vision backbones (ResNet · EfficientNet · ViT · DINOv2), 2 calibrated text encoders,
          and CLIP zero-shot search — served live from fine-tuned models on the Hugging Face Hub.
        </p>

        <div className="flex gap-6 mt-6 border-b hairline">
          {TABS.map((t) => (
            <button key={t} onClick={() => setTab(t)} data-active={tab === t}
              className="tab pb-2.5 text-sm font-medium">{t}</button>
          ))}
        </div>
      </header>

      <main className="flex-1 pb-12">
        {/* Per-tab plain-language explanation banner */}
        {tab !== "About" && (
          <div className="panel px-4 py-3 mb-5 flex items-start gap-2.5" style={{ background: "var(--panel-2)" }}>
            <span className="mono text-xs mt-0.5" style={{ color: "var(--accent)" }}>▸</span>
            <p className="text-sm" style={{ color: "var(--muted)" }}>{TAB_DESC[tab]}</p>
          </div>
        )}

        {tab === "About" ? (
          <AboutPanel />
        ) : status === "error" ? (
          <div className="panel p-6 text-sm" style={{ color: "var(--muted)" }}>
            Couldn&apos;t reach the inference API. The free Space may be waking from sleep — refresh in ~30s, or{" "}
            <a href={HF_SPACE} target="_blank" rel="noopener noreferrer" className="underline" style={{ color: "var(--accent)" }}>open the Space</a> to wake it.
          </div>
        ) : status === "connecting" ? (
          <div className="panel p-10 flex justify-center"><Spinner label="connecting to the inference API on Hugging Face…" /></div>
        ) : models && results ? (
          <>
            {tab === "Vision" && <VisionPanel models={models.vision} />}
            {tab === "Text" && <TextPanel models={models.text} />}
            {tab === "CLIP Search" && <ClipPanel />}
            {tab === "Results" && <ResultsPanel data={results} />}
          </>
        ) : null}
      </main>

      <footer className="py-5 border-t hairline mono text-[11px] flex flex-wrap gap-x-4 gap-y-1" style={{ color: "var(--faint)" }}>
        <span>Next.js + Vercel → FastAPI on HF Space → 7 models on HF Hub</span>
        <span>EuroSAT · dair-ai/emotion · CLIP ViT-B/32</span>
      </footer>
    </div>
  );
}

function StatusBadge({ status }: { status: "connecting" | "live" | "error" }) {
  const map = {
    connecting: { c: "var(--warn)", t: "connecting" },
    live: { c: "var(--accent)", t: "live" },
    error: { c: "var(--danger)", t: "waking" },
  }[status];
  return (
    <span className="chip flex items-center gap-1.5">
      <span className="w-1.5 h-1.5 rounded-full live-dot" style={{ background: map.c }} />
      {map.t}
    </span>
  );
}
