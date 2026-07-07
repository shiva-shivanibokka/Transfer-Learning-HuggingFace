"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ResultsPayload } from "@/lib/api";
import { Help, Panel, SectionTitle } from "@/components/ui";

const SERIES = [
  { key: "ResNet-50", color: "#22d3ee" },
  { key: "EfficientNet-B0", color: "#a78bfa" },
  { key: "ViT-Base", color: "#34d399" },
  { key: "DINOv2-Base", color: "#f59e0b" },
];

function td(v: string | number, hi = false) {
  return <td className="mono px-3 py-1.5 text-right" style={{ color: hi ? "var(--accent)" : "var(--fg)" }}>{v}</td>;
}

export default function ResultsPanel({ data }: { data: ResultsPayload }) {
  const de = ["p1", "p5", "p10", "p100"].map((p, i) => {
    const row: Record<string, number | string> = { pct: ["1%", "5%", "10%", "100%"][i] };
    for (const m of data.vision_data_efficiency) row[m.model as string] = m[p] as number;
    return row;
  });

  return (
    <div className="space-y-5">
      {/* Findings */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="mono text-[11px] tracking-widest uppercase" style={{ color: "var(--accent)" }}>Key findings</div>
          <Help text="The three headline takeaways from the experiment grid — the answers to the project's research questions, backed by the tables below." />
        </div>
        <div className="grid md:grid-cols-3 gap-3">
          {data.findings.map((f, i) => (
            <Panel key={i} className="!p-4">
              <div className="mono text-2xl" style={{ color: ["var(--accent)", "var(--accent2)", "#a78bfa"][i] }}>0{i + 1}</div>
              <p className="text-sm mt-1.5" style={{ color: "var(--muted)" }}>{f}</p>
            </Panel>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        {/* Strategy comparison */}
        <Panel>
          <SectionTitle kicker="EuroSAT · 100% data" title="Fine-tuning strategy comparison"
            help="Test accuracy for each backbone under three transfer strategies: linear probe (backbone frozen, train only the head), partial unfreeze (last blocks), and full fine-tune. CPU ms is single-image latency." />
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: "var(--faint)" }} className="mono uppercase text-[10px]">
                  <th className="text-left px-3 py-1.5">Model</th>
                  <th className="text-right px-3 py-1.5">Linear</th>
                  <th className="text-right px-3 py-1.5">Partial</th>
                  <th className="text-right px-3 py-1.5">Full</th>
                  <th className="text-right px-3 py-1.5">CPU ms</th>
                </tr>
              </thead>
              <tbody>
                {data.vision_strategy.map((r) => (
                  <tr key={r.model as string} className="border-t hairline">
                    <td className="px-3 py-1.5">{r.model} <span style={{ color: "var(--faint)" }}>· {r.family}</span></td>
                    {td(r.linear_probe as number, r.model === "DINOv2-Base")}
                    {td(r.partial_unfreeze as number)}
                    {td(r.full_finetune as number)}
                    {td(r.latency_ms as number)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs mt-3" style={{ color: "var(--muted)" }}>DINOv2 linear probe (95.4%) ≈ others fully fine-tuned — frozen self-supervised features transfer best.</p>
        </Panel>

        {/* Data efficiency chart */}
        <Panel>
          <SectionTitle kicker="full fine-tune" title="Data efficiency"
            help="Test accuracy as the labeled training set shrinks to 10 / 5 / 1%, using full fine-tuning. Shows which backbones stay accurate with little data." />
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={de} margin={{ top: 8, right: 14, bottom: 20, left: 8 }}>
                <CartesianGrid stroke="#23262d" vertical={false} />
                <XAxis dataKey="pct" tick={{ fill: "#8a909c", fontSize: 11 }} stroke="#23262d"
                  label={{ value: "% of training data", position: "insideBottom", offset: -10, fill: "#8a909c", fontSize: 11 }} />
                <YAxis domain={[20, 100]} tick={{ fill: "#8a909c", fontSize: 11 }} stroke="#23262d"
                  label={{ value: "Test accuracy (%)", angle: -90, position: "insideLeft", offset: 18, fill: "#8a909c", fontSize: 11, style: { textAnchor: "middle" } }} />
                <Tooltip contentStyle={{ background: "#16181d", border: "1px solid #2f333c", borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: "#e6e8ec" }} />
                {SERIES.map((s) => (
                  <Line key={s.key} type="monotone" dataKey={s.key} stroke={s.color} strokeWidth={2} dot={{ r: 2 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap gap-3 mt-2">
            {SERIES.map((s) => (
              <div key={s.key} className="flex items-center gap-1.5 text-xs" style={{ color: "var(--muted)" }}>
                <span className="w-2.5 h-0.5 rounded" style={{ background: s.color }} />{s.key}
              </div>
            ))}
          </div>
          <p className="text-xs mt-2" style={{ color: "var(--muted)" }}>ViT hits 90.5% on 1% data; DINOv2 full-tune collapses to 29% (overfits) — freeze it instead.</p>
        </Panel>

        {/* Text calibration */}
        <Panel>
          <SectionTitle kicker="dair-ai/emotion" title="Text models + calibration"
            help="Emotion-classification accuracy and F1, plus Expected Calibration Error (ECE) before and after temperature scaling. Lower ECE = the model's confidence better matches its accuracy. T is the fitted temperature." />
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: "var(--faint)" }} className="mono uppercase text-[10px]">
                  <th className="text-left px-3 py-1.5">Model</th>
                  <th className="text-right px-3 py-1.5">Acc</th>
                  <th className="text-right px-3 py-1.5">F1</th>
                  <th className="text-right px-3 py-1.5">ECE↓ before</th>
                  <th className="text-right px-3 py-1.5">after</th>
                  <th className="text-right px-3 py-1.5">T</th>
                </tr>
              </thead>
              <tbody>
                {data.text.map((r) => (
                  <tr key={r.model as string} className="border-t hairline">
                    <td className="px-3 py-1.5">{r.model}</td>
                    {td(r.accuracy as number)}
                    {td(r.f1_macro as number)}
                    {td(r.ece_before as number)}
                    {td(r.ece_after as number, true)}
                    {td(r.temperature as number)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs mt-3" style={{ color: "var(--muted)" }}>Temperature scaling lowers calibration error (ECE) in every case.</p>
        </Panel>

        {/* CLIP prompts */}
        <Panel>
          <SectionTitle kicker="CLIP zero-shot" title="Prompt sensitivity"
            help="CLIP classifies EuroSAT with no training by comparing images to text prompts. Accuracy varies a lot with prompt wording; averaging embeddings across all five templates (Ensemble) beats any single one." />
          <div className="space-y-1.5">
            {data.clip_prompts.map((p) => {
              const w = (p.accuracy / 55) * 100;
              return (
                <div key={p.template} className="grid grid-cols-[1fr_44px] items-center gap-2">
                  <div className="h-6 rounded relative overflow-hidden" style={{ background: "var(--panel-2)" }}>
                    <div className="h-full rounded" style={{ width: `${w}%`, background: p.ensemble ? "var(--accent)" : "color-mix(in srgb, var(--accent2) 55%, transparent)" }} />
                    <span className="absolute left-2 top-1/2 -translate-y-1/2 text-[11px] truncate max-w-[90%]" title={p.template}>{p.template}</span>
                  </div>
                  <span className="mono text-[11px] text-right" style={{ color: p.ensemble ? "var(--accent)" : "var(--muted)" }}>{p.accuracy.toFixed(1)}</span>
                </div>
              );
            })}
          </div>
          <p className="text-xs mt-3" style={{ color: "var(--muted)" }}>Domain-aware prompts beat generic by ~10 pts; ensembling recovers to 53.1%.</p>
        </Panel>
      </div>
    </div>
  );
}
