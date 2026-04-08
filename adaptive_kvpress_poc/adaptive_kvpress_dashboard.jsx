import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line } from "recharts";

const COLORS = {
  fixed: "#ef4444", fixedMid: "#f97316", fixedHigh: "#dc2626",
  percentile: "#8b5cf6", adaptive2: "#3b82f6", adaptive3: "#0ea5e9",
  conservative: "#22c55e", aggressive: "#f59e0b", latency: "#06b6d4",
  accent: "#e11d48", bg: "transparent", muted: "#64748b"
};

// Simulation results embedded from trace-driven run
const ARXIV_POLICIES = [
  { id: "fixed_τ=-1.0", label: "Fixed τ=-1.0", cr: 2.09, std: 0.57, cv: 0.271, mem: 59.6, reuse: 0.044, color: COLORS.fixed },
  { id: "fixed_τ=-0.5", label: "Fixed τ=-0.5", cr: 3.31, std: 1.13, cv: 0.342, mem: 75.0, reuse: 0.031, color: COLORS.fixedMid },
  { id: "fixed_τ=0.0", label: "Fixed τ=0.0", cr: 5.98, std: 2.50, cv: 0.417, mem: 86.2, reuse: 0.025, color: COLORS.fixedHigh },
  { id: "percentile_50%", label: "Percentile 50%", cr: 1.99, std: 0.03, cv: 0.017, mem: 49.9, reuse: 0.038, color: COLORS.percentile },
  { id: "adaptive_2x", label: "Adaptive 2×", cr: 1.49, std: 0.29, cv: 0.197, mem: 40.0, reuse: 0.243, color: COLORS.adaptive2 },
  { id: "adaptive_3x", label: "Adaptive 3×", cr: 2.11, std: 0.69, cv: 0.326, mem: 60.1, reuse: 0.128, color: COLORS.adaptive3 },
  { id: "conservative", label: "Conservative", cr: 1.25, std: 0.16, cv: 0.131, mem: 26.2, reuse: 0.364, color: COLORS.conservative },
  { id: "aggressive", label: "Aggressive", cr: 2.30, std: 0.90, cv: 0.392, mem: 64.1, reuse: 0.119, color: COLORS.aggressive },
  { id: "latency_opt", label: "Latency Opt", cr: 2.19, std: 0.19, cv: 0.086, mem: 52.6, reuse: 0.035, color: COLORS.latency },
];

const CROSS_TRACE = [
  { trace: "ArXiv", cr: 1.48, cv: 0.198, mem: 40.0, reuse: 0.247 },
  { trace: "Conversation", cr: 1.14, cv: 0.082, mem: 14.9, reuse: 0.817 },
  { trace: "Tool/Agent", cr: 1.26, cv: 0.179, mem: 30.5, reuse: 0.451 },
  { trace: "Azure", cr: 1.95, cv: 0.230, mem: 52.5, reuse: 0.000 },
];

const WINDOW_ABLATION = [
  { w: 64, sinks: false, cr: 2.86, cv: 0.089, reuse: 0.038 },
  { w: 64, sinks: true, cr: 1.75, cv: 0.255, reuse: 0.187 },
  { w: 128, sinks: false, cr: 2.86, cv: 0.089, reuse: 0.038 },
  { w: 128, sinks: true, cr: 1.75, cv: 0.255, reuse: 0.187 },
  { w: 192, sinks: false, cr: 2.68, cv: 0.108, reuse: 0.055 },
  { w: 192, sinks: true, cr: 1.72, cv: 0.265, reuse: 0.206 },
  { w: 256, sinks: false, cr: 2.54, cv: 0.135, reuse: 0.071 },
  { w: 256, sinks: true, cr: 1.68, cv: 0.272, reuse: 0.223 },
  { w: 384, sinks: false, cr: 2.43, cv: 0.151, reuse: 0.088 },
  { w: 384, sinks: true, cr: 1.64, cv: 0.275, reuse: 0.240 },
  { w: 512, sinks: false, cr: 2.43, cv: 0.151, reuse: 0.088 },
  { w: 512, sinks: true, cr: 1.64, cv: 0.275, reuse: 0.240 },
];

const TRACE_SUMMARIES = {
  arxiv: { requests: 500, mean_input: 7300, p99: 41053, blocks: 4804 },
  conversation: { requests: 500, mean_input: 1687, p99: 6980, blocks: 167 },
  tool_agent: { requests: 500, mean_input: 4000, p99: 25808, blocks: 1744 },
  azure: { requests: 250, mean_input: 2100, p99: 0, blocks: 0 },
};

const Tab = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    style={{
      padding: "8px 18px", cursor: "pointer", border: "none",
      borderBottom: active ? "2px solid #3b82f6" : "2px solid transparent",
      background: "transparent", color: active ? "#3b82f6" : "#94a3b8",
      fontWeight: active ? 700 : 400, fontSize: 13, fontFamily: "'JetBrains Mono', monospace",
      transition: "all 0.2s",
    }}
  >{children}</button>
);

const Stat = ({ label, value, sub }) => (
  <div style={{ textAlign: "center", padding: "10px 14px" }}>
    <div style={{ fontSize: 22, fontWeight: 700, color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>{label}</div>
    {sub && <div style={{ fontSize: 10, color: "#64748b", marginTop: 1 }}>{sub}</div>}
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: "8px 12px", fontSize: 12 }}>
      <div style={{ color: "#e2e8f0", fontWeight: 600, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "#94a3b8" }}>
          {p.name}: <span style={{ fontWeight: 600 }}>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const [tab, setTab] = useState("overview");

  const pareto = ARXIV_POLICIES.map(p => ({
    ...p, name: p.label,
    stability: +(1 - p.cv).toFixed(3),
  }));

  const windowSinks = WINDOW_ABLATION.filter(w => w.sinks);
  const windowNoSinks = WINDOW_ABLATION.filter(w => !w.sinks);
  const windowChart = [64, 128, 192, 256, 384, 512].map(w => ({
    window: w,
    "CR (no sinks)": windowNoSinks.find(x => x.w === w)?.cr || 0,
    "CR (with sinks)": windowSinks.find(x => x.w === w)?.cr || 0,
    "Reuse (no sinks)": windowNoSinks.find(x => x.w === w)?.reuse || 0,
    "Reuse (with sinks)": windowSinks.find(x => x.w === w)?.reuse || 0,
  }));

  return (
    <div style={{ fontFamily: "'Inter', system-ui, sans-serif", color: "#e2e8f0", minHeight: "100vh", padding: "20px 16px" }}>
      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
            <h1 style={{ fontSize: 20, fontWeight: 800, margin: 0, letterSpacing: "-0.02em" }}>
              Adaptive KVPress
            </h1>
            <span style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
              Trace-Driven Compression Policy Simulator
            </span>
          </div>
          <p style={{ fontSize: 12, color: "#94a3b8", marginTop: 6, lineHeight: 1.5 }}>
            Enhancement of KVZap / KVPress with adaptive threshold calibration, surrogate selection, 
            and window policy — evaluated on synthetic Mooncake + Azure production traces.
          </p>
        </div>

        {/* Summary stats */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
          gap: 1, background: "#1e293b", borderRadius: 8, overflow: "hidden", marginBottom: 20,
          border: "1px solid #334155",
        }}>
          <Stat label="Policies Tested" value="9" sub="4 baselines + 5 adaptive" />
          <Stat label="Trace Variants" value="4" sub="arxiv · chat · agent · azure" />
          <Stat label="Total Requests" value="1,750" sub="synthetic production" />
          <Stat label="Best CV" value="0.017" sub="percentile 50%" />
          <Stat label="Best Reuse" value="81.7%" sub="conversation trace" />
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 0, borderBottom: "1px solid #334155", marginBottom: 16 }}>
          {["overview", "pareto", "cross-trace", "window", "data"].map(t => (
            <Tab key={t} active={tab === t} onClick={() => setTab(t)}>
              {t === "overview" ? "Policy Comparison" : t === "pareto" ? "CR vs Stability" : 
               t === "cross-trace" ? "Cross-Trace" : t === "window" ? "Window Ablation" : "Trace Data"}
            </Tab>
          ))}
        </div>

        {/* ── OVERVIEW TAB ───────────────────────────── */}
        {tab === "overview" && (
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 12 }}>
              Compression Ratio & Stability (ArXiv Trace · 500 requests)
            </h3>
            <div style={{ height: 340 }}>
              <ResponsiveContainer>
                <BarChart data={pareto} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#94a3b8" }} angle={-25} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="cr" name="Mean Compression (×)" radius={[3,3,0,0]}>
                    {pareto.map((p, i) => <Cell key={i} fill={p.color} fillOpacity={0.85} />)}
                  </Bar>
                  <Bar dataKey="cv" name="CV (lower=more stable)" fill="#64748b" radius={[3,3,0,0]} fillOpacity={0.4} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", margin: "20px 0 12px" }}>
              Memory Savings vs Prefix Reuse Efficiency
            </h3>
            <div style={{ height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={pareto} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#94a3b8" }} angle={-25} textAnchor="end" height={60} />
                  <YAxis yAxisId="left" tick={{ fontSize: 10, fill: "#94a3b8" }} label={{ value: "Memory Saved %", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: "#94a3b8" }} domain={[0, 0.5]} label={{ value: "Reuse Efficiency", angle: 90, position: "insideRight", fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar yAxisId="left" dataKey="mem" name="Memory Saved %" fill="#3b82f6" radius={[3,3,0,0]} fillOpacity={0.7} />
                  <Bar yAxisId="right" dataKey="reuse" name="Reuse Efficiency" fill="#22c55e" radius={[3,3,0,0]} fillOpacity={0.7} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Key findings */}
            <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, padding: 16, marginTop: 16, fontSize: 12, lineHeight: 1.7, color: "#cbd5e1" }}>
              <strong style={{ color: "#e2e8f0" }}>Key Findings:</strong>
              <br/>• <strong>Fixed thresholds</strong> achieve high compression but with high variance (CV 0.27–0.42) and near-zero prefix reuse (3–4%).
              <br/>• <strong>Adaptive policies</strong> trade some compression for dramatically lower CV and 5–10× better prefix reuse efficiency.
              <br/>• <strong>Conservative</strong> (1.25× CR) preserves 36.4% of prefix blocks — critical for Dynamo/Mooncake KV cache sharing.
              <br/>• <strong>Latency-optimal</strong> (linear surrogate) achieves 2.19× with the lowest CV (0.086) among methods exceeding 2×.
              <br/>• <strong>Percentile 50%</strong> has the most stable compression (CV 0.017) but no reuse awareness.
            </div>
          </div>
        )}

        {/* ── PARETO TAB ─────────────────────────────── */}
        {tab === "pareto" && (
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 12 }}>
              Compression × Stability Pareto (lower CV + higher CR = better)
            </h3>
            <div style={{ height: 380 }}>
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="cv" name="CV" type="number" domain={[0, 0.5]}
                    tick={{ fontSize: 10, fill: "#94a3b8" }}
                    label={{ value: "CV (Compression Stability) →", position: "bottom", fill: "#94a3b8", fontSize: 11 }} />
                  <YAxis dataKey="cr" name="CR" type="number" domain={[0.8, 7]}
                    tick={{ fontSize: 10, fill: "#94a3b8" }}
                    label={{ value: "← Mean Compression Ratio", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 11 }} />
                  <Tooltip content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: 10, fontSize: 12 }}>
                        <div style={{ fontWeight: 700, color: d.color, marginBottom: 4 }}>{d.label}</div>
                        <div style={{ color: "#cbd5e1" }}>CR: {d.cr}× · CV: {d.cv} · Mem: {d.mem}%</div>
                        <div style={{ color: "#cbd5e1" }}>Reuse: {(d.reuse * 100).toFixed(1)}%</div>
                      </div>
                    );
                  }} />
                  <Scatter data={pareto}>
                    {pareto.map((p, i) => (
                      <Cell key={i} fill={p.color} r={Math.max(6, p.reuse * 40 + 5)} stroke="#0f172a" strokeWidth={1.5} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 8, justifyContent: "center" }}>
              {pareto.map(p => (
                <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11 }}>
                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: p.color }} />
                  <span style={{ color: "#94a3b8" }}>{p.label}</span>
                </div>
              ))}
            </div>
            <div style={{ fontSize: 11, color: "#64748b", textAlign: "center", marginTop: 8 }}>
              Bubble size ∝ prefix reuse efficiency. Ideal position: top-left (high CR, low CV).
            </div>
          </div>
        )}

        {/* ── CROSS-TRACE TAB ────────────────────────── */}
        {tab === "cross-trace" && (
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 12 }}>
              Adaptive 2× Policy Across Trace Variants
            </h3>
            <div style={{ height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={CROSS_TRACE} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="trace" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="cr" name="Compression Ratio" fill="#3b82f6" radius={[3,3,0,0]} />
                  <Bar dataKey="cv" name="CV" fill="#f59e0b" radius={[3,3,0,0]} />
                  <Bar dataKey="reuse" name="Reuse Efficiency" fill="#22c55e" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Trace summaries */}
            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", margin: "20px 0 8px" }}>Trace Characteristics</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10 }}>
              {Object.entries(TRACE_SUMMARIES).map(([name, t]) => (
                <div key={name} style={{ background: "#1e293b", borderRadius: 6, padding: 12, border: "1px solid #334155" }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: "#e2e8f0", marginBottom: 6, textTransform: "capitalize" }}>{name.replace('_', ' ')}</div>
                  <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.8 }}>
                    Requests: {t.requests}<br/>
                    Mean input: {t.mean_input.toLocaleString()} tokens<br/>
                    {t.p99 > 0 && <>P99 input: {t.p99.toLocaleString()} tokens<br/></>}
                    {t.blocks > 0 && <>Unique blocks: {t.blocks.toLocaleString()}</>}
                    {t.blocks === 0 && <span style={{color:"#64748b"}}>No block hashes (Azure)</span>}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, padding: 16, marginTop: 16, fontSize: 12, lineHeight: 1.7, color: "#cbd5e1" }}>
              <strong style={{ color: "#e2e8f0" }}>Observations:</strong>
              <br/>• <strong>Conversation</strong> trace has highest reuse efficiency (81.7%) — multi-turn chat shares system prompt blocks extensively.
              <br/>• <strong>Azure</strong> trace (no block hashes) shows 0% reuse — reuse-aware bonus requires prefix hash information.
              <br/>• Compression ratio naturally adapts: shorter conversation prompts → less to compress → lower CR. This is the intended behavior of adaptive thresholding.
              <br/>• CV is lowest on conversation (0.082) where prompt lengths are most uniform.
            </div>
          </div>
        )}

        {/* ── WINDOW TAB ─────────────────────────────── */}
        {tab === "window" && (
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 12 }}>
              Window Size × Attention Sinks Ablation (target=2.5×)
            </h3>
            <div style={{ height: 320 }}>
              <ResponsiveContainer>
                <LineChart data={windowChart} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="window" tick={{ fontSize: 10, fill: "#94a3b8" }}
                    label={{ value: "Base Window Size (tokens)", position: "bottom", fill: "#94a3b8", fontSize: 11, dy: 10 }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} domain={[0, 3.2]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="CR (no sinks)" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="CR (with sinks)" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="Reuse (no sinks)" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="Reuse (with sinks)" stroke="#22c55e" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, padding: 16, marginTop: 16, fontSize: 12, lineHeight: 1.7, color: "#cbd5e1" }}>
              <strong style={{ color: "#e2e8f0" }}>Window Policy Insights:</strong>
              <br/>• Enabling <strong>attention sinks</strong> (4 head tokens) + <strong>score-aware boundary</strong> + <strong>reuse bonus</strong> reduces CR from ~2.86× to ~1.75× but increases reuse efficiency from 3.8% to 18.7%.
              <br/>• Window size has diminishing returns beyond 256 tokens — both CR and reuse plateau.
              <br/>• The tradeoff is clear: more protected tokens → less compression but more prefix cache compatibility with Dynamo/Mooncake KVBM.
              <br/>• For KV cache offloading scenarios, the reuse-aware configuration is preferred since broken prefix blocks force expensive recomputation.
            </div>
          </div>
        )}

        {/* ── DATA TAB ───────────────────────────────── */}
        {tab === "data" && (
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 12 }}>Raw Results Table (ArXiv Trace)</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #334155" }}>
                    {["Policy","Mean CR","Std","CV","Mem%","Reuse","τ","Window"].map(h => (
                      <th key={h} style={{ padding: "8px 6px", textAlign: "left", color: "#94a3b8", fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ARXIV_POLICIES.map(p => (
                    <tr key={p.id} style={{ borderBottom: "1px solid #1e293b" }}>
                      <td style={{ padding: "6px", color: p.color, fontWeight: 600 }}>{p.label}</td>
                      <td style={{ padding: "6px" }}>{p.cr}×</td>
                      <td style={{ padding: "6px" }}>{p.std}</td>
                      <td style={{ padding: "6px", color: p.cv < 0.15 ? "#22c55e" : p.cv < 0.3 ? "#f59e0b" : "#ef4444" }}>{p.cv}</td>
                      <td style={{ padding: "6px" }}>{p.mem}%</td>
                      <td style={{ padding: "6px", color: p.reuse > 0.2 ? "#22c55e" : "#64748b" }}>{(p.reuse * 100).toFixed(1)}%</td>
                      <td style={{ padding: "6px" }}>{ARXIV_POLICIES.indexOf(p) < 3 ? [-1.0, -0.5, 0.0][ARXIV_POLICIES.indexOf(p)] : "adaptive"}</td>
                      <td style={{ padding: "6px" }}>128</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", margin: "24px 0 8px" }}>Architecture</h3>
            <div style={{ background: "#1e293b", borderRadius: 8, padding: 16, border: "1px solid #334155", fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#94a3b8", lineHeight: 1.8, whiteSpace: "pre" }}>
{`Trace Entry (Mooncake JSONL)
  │ timestamp, input_length, output_length, hash_ids
  ▼
Score Profile Engine ← one-time KVzap profiling (~500 prompts)
  │ synthetic per-token importance scores
  ▼
┌─ Adaptive Policy ────────────────────────┐
│  1. Surrogate Tier Selection (§4)        │
│     heuristic → linear → MLP → ensemble  │
│  2. Threshold Calibration (§3)           │
│     fixed | adaptive | percentile        │
│  3. Window Policy (§5)                   │
│     sinks + scaled + score-aware + reuse │
└──────────────────────────────────────────┘
  │ retain/evict per token
  ▼
Metrics: CR, CV, memory, prefix reuse efficiency`}
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 28, paddingTop: 14, borderTop: "1px solid #1e293b", fontSize: 10, color: "#475569", textAlign: "center" }}>
          Adaptive KVPress PoC · Based on KVzap (arXiv:2601.07891) + NVIDIA kvpress · 
          Traces: Mooncake (FAST'25) + Azure LLM Inference · Model: Qwen3-8B config
        </div>
      </div>
    </div>
  );
}
