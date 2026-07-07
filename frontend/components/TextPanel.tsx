"use client";

import { useState } from "react";
import { api, type TextModel, type TextResult } from "@/lib/api";
import { ErrorNote, Metric, Panel, ProbBars, SectionTitle, Spinner } from "@/components/ui";

const EXAMPLES = [
  "I can't stop crying, everything feels hopeless.",
  "Best day ever — I just got the offer!",
  "omg I'm so nervous about the interview tomorrow",
  "I really love spending time with my family",
];

export default function TextPanel({ models }: { models: TextModel[] }) {
  const [model, setModel] = useState(models[0]?.key);
  const [text, setText] = useState("");
  const [res, setRes] = useState<TextResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const meta = models.find((m) => m.key === model);

  async function detect() {
    if (!text.trim() || !model) return;
    setLoading(true); setErr(null); setRes(null);
    try {
      setRes(await api.text(text, model));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <Panel>
        <SectionTitle kicker="dair-ai/emotion · 6 classes" title="Emotion detector"
          sub="Fine-tuned encoder + temperature-scaled calibration." />

        <div className="flex flex-wrap gap-2 mb-3">
          {models.map((m) => (
            <button key={m.key} onClick={() => { setModel(m.key); setRes(null); }}
              className="chip" data-active={m.key === model}
              style={m.key === model ? { borderColor: "var(--accent)", color: "var(--fg)" } : {}}>
              {m.key}
            </button>
          ))}
        </div>

        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4}
          placeholder="Type a sentence or a tweet…"
          className="input w-full p-3 text-sm resize-none" />

        <div className="flex flex-wrap gap-1.5 mt-3">
          {EXAMPLES.map((ex) => (
            <button key={ex} onClick={() => { setText(ex); setRes(null); }}
              className="chip text-left max-w-full truncate" title={ex}
              style={{ maxWidth: 220 }}>{ex}</button>
          ))}
        </div>

        <button onClick={detect} disabled={!text.trim() || loading}
          className="btn-accent w-full mt-4 py-2.5 text-sm">
          {loading ? "Analyzing…" : "Detect emotion"}
        </button>
      </Panel>

      <Panel>
        <SectionTitle kicker={meta ? `${meta.params_m}M params · ${meta.year}` : ""} title="Prediction + calibration" />
        {!res && !loading && !err && (
          <div className="text-sm py-16 text-center" style={{ color: "var(--faint)" }}>
            Raw vs. temperature-scaled confidence will appear side by side.
          </div>
        )}
        {loading && <div className="py-16 flex justify-center"><Spinner label="inferring…" /></div>}
        {err && <ErrorNote msg={err} />}
        {res && (
          <div className="space-y-4">
            <div className="flex items-end justify-between">
              <div>
                <div className="mono text-[10px] uppercase tracking-wider" style={{ color: "var(--faint)" }}>emotion</div>
                <div className="text-2xl font-semibold uppercase" style={{ color: "var(--accent)" }}>{res.label}</div>
              </div>
              <div className="mono text-3xl">{(res.confidence * 100).toFixed(1)}<span className="text-lg" style={{ color: "var(--muted)" }}>%</span></div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="mono text-[10px] uppercase tracking-wider mb-2" style={{ color: "var(--danger)" }}>raw</div>
                <ProbBars items={res.raw} topN={6} color="var(--danger)" />
              </div>
              <div>
                <div className="mono text-[10px] uppercase tracking-wider mb-2" style={{ color: "var(--accent)" }}>calibrated · T={res.temperature.toFixed(3)}</div>
                <ProbBars items={res.calibrated} topN={6} color="var(--accent)" />
              </div>
            </div>
            <Metric label="temperature (scaling)" value={res.temperature.toFixed(3)} accent="var(--accent2)" />
          </div>
        )}
      </Panel>
    </div>
  );
}
