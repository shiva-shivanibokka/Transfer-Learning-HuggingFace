"use client";

import { useState } from "react";
import { api, type ClipHit } from "@/lib/api";
import { ErrorNote, Panel, SectionTitle, Spinner } from "@/components/ui";

const EXAMPLES = ["a river through farmland", "dense green forest", "urban residential blocks", "a busy highway", "industrial buildings", "crops from above"];

export default function ClipPanel() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(8);
  const [hits, setHits] = useState<ClipHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function search(q?: string) {
    const qq = (q ?? query).trim();
    if (!qq) return;
    setQuery(qq);
    setLoading(true); setErr(null); setHits(null);
    try {
      const r = await api.clip(qq, k);
      setHits(r.results);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Panel>
      <SectionTitle kicker="CLIP ViT-B/32 · 1000 EuroSAT images"
        title="Text → image retrieval"
        sub="Zero-shot search over a cached embedding index using natural language." />

      <div className="flex gap-2">
        <input value={query} onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
          placeholder="a satellite image of…"
          className="input flex-1 px-3 py-2.5 text-sm" />
        <select value={k} onChange={(e) => setK(Number(e.target.value))}
          className="input px-2 text-sm mono">
          {[4, 8, 12].map((n) => <option key={n} value={n}>k={n}</option>)}
        </select>
        <button onClick={() => search()} disabled={!query.trim() || loading}
          className="btn-accent px-5 text-sm">Search</button>
      </div>

      <div className="flex flex-wrap gap-1.5 mt-3">
        {EXAMPLES.map((ex) => (
          <button key={ex} onClick={() => search(ex)} className="chip">{ex}</button>
        ))}
      </div>

      <div className="mt-5">
        {loading && <div className="py-16 flex justify-center"><Spinner label="encoding query + ranking 1000 images…" /></div>}
        {err && <ErrorNote msg={err} />}
        {hits && hits.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {hits.map((h, i) => (
              <div key={i} className="panel p-0 overflow-hidden" style={{ background: "var(--panel-2)" }}>
                <img src={h.image_png} alt={h.label} className="w-full aspect-square object-cover" />
                <div className="px-2 py-1.5 flex items-center justify-between">
                  <span className="text-xs truncate" title={h.label}>{h.label}</span>
                  <span className="mono text-[10px]" style={{ color: "var(--accent)" }}>{h.similarity.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
        {!loading && !hits && !err && (
          <div className="text-sm py-16 text-center" style={{ color: "var(--faint)" }}>
            Retrieved patches appear here, ranked by cosine similarity.
          </div>
        )}
      </div>
    </Panel>
  );
}
