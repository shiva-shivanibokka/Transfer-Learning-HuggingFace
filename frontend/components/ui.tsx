"use client";

import type { Prob } from "@/lib/api";

export function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`panel p-5 ${className}`}>{children}</div>;
}

export function Help({ text }: { text: string }) {
  return (
    <span className="help" aria-label={text} role="img">
      <span className="dot mono">?</span>
      <span className="tip">{text}</span>
    </span>
  );
}

export function SectionTitle({ kicker, title, sub, help }: { kicker?: string; title: string; sub?: string; help?: string }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-3">
      <div>
        {kicker && <div className="mono text-[11px] tracking-widest uppercase" style={{ color: "var(--accent)" }}>{kicker}</div>}
        <h2 className="text-lg font-semibold mt-1">{title}</h2>
        {sub && <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>{sub}</p>}
      </div>
      {help && <div className="shrink-0 mt-0.5"><Help text={help} /></div>}
    </div>
  );
}

export function Metric({ label, value, unit, accent }: { label: string; value: string | number; unit?: string; accent?: string }) {
  return (
    <div className="panel px-4 py-3" style={{ background: "var(--panel-2)" }}>
      <div className="mono text-[10px] uppercase tracking-wider" style={{ color: "var(--faint)" }}>{label}</div>
      <div className="mono text-xl mt-1" style={{ color: accent || "var(--fg)" }}>
        {value}
        {unit && <span className="text-sm ml-1" style={{ color: "var(--muted)" }}>{unit}</span>}
      </div>
    </div>
  );
}

// Native horizontal probability bars — the top prediction highlighted.
export function ProbBars({ items, topN = 6, color = "var(--accent)" }: { items: Prob[]; topN?: number; color?: string }) {
  const sorted = [...items].sort((a, b) => b.prob - a.prob).slice(0, topN);
  const max = Math.max(...sorted.map((s) => s.prob), 0.001);
  return (
    <div className="space-y-1.5">
      {sorted.map((it, i) => (
        <div key={it.label} className="grid grid-cols-[110px_1fr_46px] items-center gap-2">
          <div className="text-xs truncate" style={{ color: i === 0 ? "var(--fg)" : "var(--muted)" }} title={it.label}>{it.label}</div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--panel-2)" }}>
            <div className="h-full rounded-full transition-all duration-500"
              style={{ width: `${(it.prob / max) * 100}%`, background: i === 0 ? color : "color-mix(in srgb, var(--muted) 45%, transparent)" }} />
          </div>
          <div className="mono text-[11px] text-right" style={{ color: i === 0 ? color : "var(--muted)" }}>
            {(it.prob * 100).toFixed(1)}
          </div>
        </div>
      ))}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm" style={{ color: "var(--muted)" }}>
      <span className="inline-block w-4 h-4 rounded-full border-2 animate-spin"
        style={{ borderColor: "var(--border)", borderTopColor: "var(--accent)" }} />
      {label}
    </div>
  );
}

export function ErrorNote({ msg }: { msg: string }) {
  return (
    <div className="text-sm rounded-lg px-3 py-2" style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.3)", color: "var(--danger)" }}>
      {msg}
    </div>
  );
}
