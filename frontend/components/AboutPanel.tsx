"use client";

import { Panel, SectionTitle } from "@/components/ui";

const DEMOS = [
  { icon: "🛰️", title: "Satellite classifier", body: "Show it an aerial photo and it tells you what's on the ground — forest, city, river, farmland. Four different AI 'brains' compete so you can see which reads satellite images best.", models: ["ResNet-50", "EfficientNet-B0", "ViT-Base", "DINOv2-Base"] },
  { icon: "💬", title: "Emotion reader", body: "Type how you feel and it detects the emotion behind your words. It also demonstrates a trick called calibration that makes the AI's confidence more honest.", models: ["RoBERTa", "ModernBERT"] },
  { icon: "🔎", title: "Search by meaning", body: "Search satellite images using plain English, like a search engine — but it matches the meaning of your words, not filenames or tags.", models: ["CLIP ViT-B/32"] },
];

const STEPS = [
  { n: "1", t: "Teach the models", d: "Seven AI models were trained on a laptop GPU to specialise in satellite images and emotion in text." },
  { n: "2", t: "Publish them online", d: "The trained models were uploaded to Hugging Face — a free public home for AI models — so anyone can use them." },
  { n: "3", t: "Ask them live", d: "This website sends your input to those models in real time and shows their answers. Nothing here is faked or pre-recorded." },
];

const TAKEAWAYS = [
  "A model that taught itself from images with no labels (DINOv2) turned out to be a much better starting point than models taught with labels.",
  "Bigger isn't always better — one smaller model ran ~100× faster with almost the same accuracy, which matters for real products.",
  "Wording matters: asking the AI the same question in different words noticeably changed its answers.",
];

export default function AboutPanel() {
  return (
    <div className="space-y-5">
      <Panel>
        <SectionTitle kicker="in plain language" title="What is this?" />
        <p className="text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
          This is a hands-on showcase of <span style={{ color: "var(--fg)" }}>transfer learning</span> — the idea that an AI
          model already trained on millions of everyday images or sentences can be <span style={{ color: "var(--fg)" }}>re-used
          and lightly re-trained</span> to become an expert at a new, specialised task. Here that task is reading{" "}
          <span style={{ color: "var(--fg)" }}>satellite photos</span> and understanding the{" "}
          <span style={{ color: "var(--fg)" }}>emotion in text</span>. Everything you click runs on real AI models, live —
          it&apos;s both an interactive demo and the write-up of a small research project.
        </p>
      </Panel>

      <div>
        <div className="mono text-[11px] tracking-widest uppercase mb-2" style={{ color: "var(--accent)" }}>The three demos</div>
        <div className="grid md:grid-cols-3 gap-3">
          {DEMOS.map((d) => (
            <Panel key={d.title} className="!p-4 flex flex-col">
              <div className="text-2xl">{d.icon}</div>
              <div className="font-semibold mt-2">{d.title}</div>
              <p className="text-sm mt-1.5" style={{ color: "var(--muted)" }}>{d.body}</p>
              <div className="mt-auto pt-3">
                <div className="mono text-[10px] uppercase tracking-wider mb-1.5" style={{ color: "var(--faint)" }}>models used</div>
                <div className="flex flex-wrap gap-1.5">
                  {d.models.map((m) => <span key={m} className="chip">{m}</span>)}
                </div>
              </div>
            </Panel>
          ))}
        </div>
      </div>

      <Panel>
        <SectionTitle kicker="the pipeline" title="How it all works" />
        <div className="grid md:grid-cols-3 gap-4">
          {STEPS.map((s) => (
            <div key={s.n} className="flex gap-3">
              <div className="mono text-lg shrink-0 w-8 h-8 rounded-full flex items-center justify-center"
                style={{ border: "1px solid var(--border-hi)", color: "var(--accent)" }}>{s.n}</div>
              <div>
                <div className="text-sm font-medium">{s.t}</div>
                <p className="text-sm mt-0.5" style={{ color: "var(--muted)" }}>{s.d}</p>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid lg:grid-cols-2 gap-5">
        <Panel>
          <SectionTitle kicker="what we learned" title="Findings, in plain words" />
          <ul className="space-y-2.5">
            {TAKEAWAYS.map((t, i) => (
              <li key={i} className="flex gap-2.5 text-sm" style={{ color: "var(--muted)" }}>
                <span style={{ color: ["var(--accent)", "var(--accent2)", "#a78bfa"][i] }} className="mono">✦</span>
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel>
          <SectionTitle kicker="for the curious" title="Under the hood" />
          <div className="space-y-2 text-sm" style={{ color: "var(--muted)" }}>
            <Row k="Frontend" v="Next.js + React, deployed on Vercel" />
            <Row k="Inference API" v="FastAPI on a Hugging Face Docker Space" />
            <Row k="Models" v="7 fine-tuned PyTorch models on the HF Hub" />
            <Row k="Vision data" v="EuroSAT — 27k Sentinel-2 satellite patches, 10 land classes" />
            <Row k="Text data" v="dair-ai/emotion — tweets labelled with 6 emotions" />
            <Row k="Backbones" v="ResNet · EfficientNet · ViT · DINOv2 · RoBERTa · ModernBERT · CLIP" />
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 border-b hairline pb-1.5">
      <span className="mono text-[11px] uppercase tracking-wider shrink-0" style={{ color: "var(--faint)" }}>{k}</span>
      <span className="text-right" style={{ color: "var(--fg)" }}>{v}</span>
    </div>
  );
}
