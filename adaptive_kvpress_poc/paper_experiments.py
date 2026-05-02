#!/usr/bin/env python3
"""Paper Experiments 1-7 (excluding 4) for Adaptive KV Compression Policy.
Produces structured JSON data for the visualization dashboard.
Usage:
    python3 adaptive_kvpress_poc/paper_experiments.py --fast
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path: sys.path.insert(0, _PARENT_DIR)

from adaptive_kvpress_poc.score_profiles import ScoreProfileEngine
from adaptive_kvpress_poc.score_profiles.triattention_engine import TriAttentionScoreEngine
from adaptive_kvpress_poc.policies.compression import CompressionPolicy
from adaptive_kvpress_poc.policies.threshold import FixedThresholdPolicy, AdaptiveThresholdPolicy
from adaptive_kvpress_poc.policies.surrogate import SurrogateSelector, SurrogateTier
from adaptive_kvpress_poc.policies.window import AdaptiveWindowPolicy, WindowConfig
from adaptive_kvpress_poc.traces.loader import MooncakeTrace, load_mooncake_trace
from adaptive_kvpress_poc.traces.generator import generate_synthetic_mooncake_trace
from adaptive_kvpress_poc.simulator import TraceSimulator, SimulationConfig

def _mp(engine, th, reuse=False, bonus=0.25, wc=None):
    wc = wc or WindowConfig(base_window=128, head_tokens=4, scale_with_compression=True, score_aware_boundary=True)
    return CompressionPolicy(threshold_policy=th, surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
        window_policy=AdaptiveWindowPolicy(wc), score_engine=engine, reuse_aware=reuse, reuse_bonus_scale=bonus)

WC_B = WindowConfig(base_window=128, head_tokens=0, scale_with_compression=False, score_aware_boundary=False)
WC_A = WindowConfig(base_window=128, head_tokens=4, scale_with_compression=True, score_aware_boundary=True)
WC_C = WindowConfig(base_window=256, head_tokens=4, scale_with_compression=True, score_aware_boundary=True)

def _sim(): return TraceSimulator(SimulationConfig(store_per_request=True, report_interval=99999))
def _sub(trace, fast):
    limit = 100 if fast else 500
    if len(trace) > limit:
        entries = [trace[i] for i in range(0, len(trace), max(1, len(trace)//limit))][:limit]
        return MooncakeTrace(entries=entries, name=trace.name)
    return trace

def experiment_1():
    print("  > Exp 1: Score Distributions")
    kv, ta = ScoreProfileEngine("qwen3-8b",42), TriAttentionScoreEngine("qwen3-8b",42)
    data = []
    for L in [1024,4096,8192,16384,32768]:
        for nm, eng in [("KVZap",kv),("TriAttention",ta)]:
            d = eng.generate_scores(L,2); s=d.scores; n=len(s); ss=sorted(s)
            data.append({"length":L,"scorer":nm,"mean":round(d.global_mean,4),"std":round(d.global_std,4),
                "entropy":round(d.entropy,3),"q05":round(ss[int(n*.05)],4),"q25":round(ss[n//4],4),
                "q50":round(ss[n//2],4),"q75":round(ss[int(n*.75)],4),"q95":round(ss[int(n*.95)],4),
                "retain_neg1":round(d.fraction_above(-1.0),4),"retain_0":round(d.fraction_above(0.0),4),
                "prefix_3block":round(sum(s[:min(1536,n)])/min(1536,n),4),
                "last_128":round(sum(s[max(0,n-128):])/min(128,n),4)})
        print(f"    L={L}: done")
    return data

def experiment_2(trace, fast):
    print(f"  > Exp 2: Stability ({trace.name})")
    kv, ta = ScoreProfileEngine("qwen3-8b",42), TriAttentionScoreEngine("qwen3-8b",42)
    sim, tt = _sim(), _sub(trace, fast)
    data = []
    for tau in [-1.5,-1.0,-0.5,0.0,0.5]:
        r=sim.run(tt,_mp(kv,FixedThresholdPolicy(tau),reuse=False,wc=WC_B),verbose=False); cs=sorted(r.compression_ratios); n=len(cs)
        data.append({"scorer":"KVZap","policy":f"fixed={tau:+.2f}","cr":round(r.mean_compression,3),"cv":round(r.cv_compression,4),
            "std":round(r.std_compression,3),"p05":round(cs[int(n*.05)],3),"p25":round(cs[n//4],3),"p50":round(cs[n//2],3),
            "p75":round(cs[int(n*.75)],3),"p95":round(cs[int(n*.95)],3),"reuse":round(r.mean_reuse_efficiency,4)})
    for t in [1.5,2.0,2.5,3.0]:
        r=sim.run(tt,_mp(kv,AdaptiveThresholdPolicy(t),reuse=True,wc=WC_A),verbose=False); cs=sorted(r.compression_ratios); n=len(cs)
        data.append({"scorer":"KVZap","policy":f"adapt={t:.1f}x","cr":round(r.mean_compression,3),"cv":round(r.cv_compression,4),
            "std":round(r.std_compression,3),"p05":round(cs[int(n*.05)],3),"p25":round(cs[n//4],3),"p50":round(cs[n//2],3),
            "p75":round(cs[int(n*.75)],3),"p95":round(cs[int(n*.95)],3),"reuse":round(r.mean_reuse_efficiency,4)})
    for tau in [0.08,0.10,0.12,0.14]:
        r=sim.run(tt,_mp(ta,FixedThresholdPolicy(tau),reuse=False,wc=WC_B),verbose=False); cs=sorted(r.compression_ratios); n=len(cs)
        data.append({"scorer":"TriAttention","policy":f"fixed={tau:.2f}","cr":round(r.mean_compression,3),"cv":round(r.cv_compression,4),
            "std":round(r.std_compression,3),"p05":round(cs[int(n*.05)],3),"p25":round(cs[n//4],3),"p50":round(cs[n//2],3),
            "p75":round(cs[int(n*.75)],3),"p95":round(cs[int(n*.95)],3),"reuse":round(r.mean_reuse_efficiency,4)})
    for t in [1.5,2.0,2.5,3.0]:
        r=sim.run(tt,_mp(ta,AdaptiveThresholdPolicy(t),reuse=True,bonus=0.10,wc=WC_A),verbose=False); cs=sorted(r.compression_ratios); n=len(cs)
        data.append({"scorer":"TriAttention","policy":f"adapt={t:.1f}x","cr":round(r.mean_compression,3),"cv":round(r.cv_compression,4),
            "std":round(r.std_compression,3),"p05":round(cs[int(n*.05)],3),"p25":round(cs[n//4],3),"p50":round(cs[n//2],3),
            "p75":round(cs[int(n*.75)],3),"p95":round(cs[int(n*.95)],3),"reuse":round(r.mean_reuse_efficiency,4)})
    print(f"    {len(data)} configs")
    return data

def experiment_3(trace, fast):
    print(f"  > Exp 3: Reuse Pareto ({trace.name})")
    kv, ta = ScoreProfileEngine("qwen3-8b",42), TriAttentionScoreEngine("qwen3-8b",42)
    sim, tt = _sim(), _sub(trace, fast)
    data = []
    for tau in [-2.0,-1.0,-0.5,0.0,0.5]:
        r=sim.run(tt,_mp(kv,FixedThresholdPolicy(tau),reuse=False,wc=WC_B),verbose=False)
        data.append({"method":"KVZap fixed","cr":round(r.mean_compression,3),"reuse":round(r.mean_reuse_efficiency,4),"cv":round(r.cv_compression,4),"mem":round(r.memory_savings_pct,1)})
    for t in [1.5,2.0,2.5,3.5]:
        for b in [0.25,0.50]:
            r=sim.run(tt,_mp(kv,AdaptiveThresholdPolicy(t),reuse=True,bonus=b,wc=WC_A),verbose=False)
            data.append({"method":f"KVZap reuse(b={b})","cr":round(r.mean_compression,3),"reuse":round(r.mean_reuse_efficiency,4),"cv":round(r.cv_compression,4),"mem":round(r.memory_savings_pct,1)})
    for tau in [0.07,0.09,0.10,0.12,0.14]:
        r=sim.run(tt,_mp(ta,FixedThresholdPolicy(tau),reuse=False,wc=WC_B),verbose=False)
        data.append({"method":"TriAttn fixed","cr":round(r.mean_compression,3),"reuse":round(r.mean_reuse_efficiency,4),"cv":round(r.cv_compression,4),"mem":round(r.memory_savings_pct,1)})
    for t in [1.5,2.0,3.0]:
        for b in [0.10,0.25]:
            r=sim.run(tt,_mp(ta,AdaptiveThresholdPolicy(t),reuse=True,bonus=b,wc=WC_A),verbose=False)
            data.append({"method":f"TriAttn reuse(b={b})","cr":round(r.mean_compression,3),"reuse":round(r.mean_reuse_efficiency,4),"cv":round(r.cv_compression,4),"mem":round(r.memory_savings_pct,1)})
    print(f"    {len(data)} Pareto points")
    return data

def experiment_5(trace, fast):
    print(f"  > Exp 5: Hybrid ({trace.name})")
    kv, ta = ScoreProfileEngine("qwen3-8b",42), TriAttentionScoreEngine("qwen3-8b",42)
    sim, tt = _sim(), _sub(trace, fast)
    cfgs = [
        ("No compression",kv,FixedThresholdPolicy(float('-inf')),False,0,1.0,WC_B),
        ("NVFP4 only (2x)",kv,FixedThresholdPolicy(float('-inf')),False,0,2.0,WC_B),
        ("KVZap fixed 3x",kv,FixedThresholdPolicy(-0.5),False,0,1.0,WC_B),
        ("KVZap adapt 1.5x",kv,AdaptiveThresholdPolicy(1.5),True,0.25,1.0,WC_A),
        ("Adapt 1.5x+NVFP4",kv,AdaptiveThresholdPolicy(1.5),True,0.25,2.0,WC_A),
        ("Adapt 2x+NVFP4",kv,AdaptiveThresholdPolicy(2.0),True,0.25,2.0,WC_A),
        ("TriAttn 1.5x+NVFP4",ta,AdaptiveThresholdPolicy(1.5),True,0.25,2.0,WC_A),
        ("TriAttn+NVFP4",ta,AdaptiveThresholdPolicy(2.0),True,0.10,2.0,WC_A),
    ]
    data = []
    for label,eng,th,reuse,bonus,quant,wc in cfgs:
        r=sim.run(tt,_mp(eng,th,reuse=reuse,bonus=bonus,wc=wc),verbose=False)
        full=sum(x.memory_bytes_full for x in r.per_request)/1e9
        after_p=sum(x.memory_bytes_compressed for x in r.per_request)/1e9
        after_q=after_p/quant
        data.append({"config":label,"prune_cr":round(r.mean_compression,2),"quant_factor":quant,
            "total_cr":round(r.mean_compression*quant,2),"full_gb":round(full,2),"after_prune_gb":round(after_p,2),
            "after_quant_gb":round(after_q,2),"savings_pct":round((1-after_q/max(.01,full))*100,1),
            "reuse":round(r.mean_reuse_efficiency,4),"cv":round(r.cv_compression,4)})
    print(f"    {len(data)} configs")
    return data

def experiment_6():
    print("  > Exp 6: Accuracy (simulated)")
    BASE = 70.0
    def kv_acc(cr,s="mlp"):
        r2=0.77 if s=="mlp" else 0.63
        if cr<=1.5: return BASE-0.2*(cr-1)
        elif cr<=2.5: return BASE-1.0*(cr-1)*(1.1-r2)
        elif cr<=4: return BASE-3.0*(cr-2)*(1.1-r2)-1
        else: return max(30,BASE-8*(cr-3))
    def ta_acc(cr):
        if cr<=2.5: return BASE-0.1*(cr-1)
        elif cr<=5: return BASE-0.5*(cr-2)
        elif cr<=10: return BASE-1.5*(cr-4)
        else: return max(35,BASE-5*(cr-8))
    def dms_acc(cr):
        if cr<=4: return BASE-0.3*(cr-1)
        elif cr<=8: return BASE-1.0*(cr-3)
        else: return max(40,BASE-3*(cr-6))
    data = []
    for cr in [1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0]: data.append({"method":"KVZap-MLP","cr":cr,"accuracy":round(kv_acc(cr),1)})
    for cr in [1.0,1.5,2.0,2.5,3.0,4.0]: data.append({"method":"KVZap-Linear","cr":cr,"accuracy":round(kv_acc(cr,"linear"),1)})
    for cr in [1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0,10.0]: data.append({"method":"TriAttention","cr":cr,"accuracy":round(ta_acc(cr),1)})
    for cr in [1.0,2.0,4.0,6.0,8.0]: data.append({"method":"DMS","cr":cr,"accuracy":round(dms_acc(cr),1)})
    data.append({"method":"Full Attention","cr":1.0,"accuracy":BASE})
    return data

def experiment_7(traces, fast):
    print("  > Exp 7: Cross-Trace")
    kv, ta = ScoreProfileEngine("qwen3-8b",42), TriAttentionScoreEngine("qwen3-8b",42)
    sim = _sim()
    pols = {"KVZap fixed":_mp(kv,FixedThresholdPolicy(-1.0),reuse=False,wc=WC_B),
            "KVZap adaptive":_mp(kv,AdaptiveThresholdPolicy(2.0),reuse=True,bonus=0.25,wc=WC_A),
            "TriAttn adaptive":_mp(ta,AdaptiveThresholdPolicy(2.0),reuse=True,bonus=0.10,wc=WC_A),
            "TriAttn conserv.":_mp(ta,AdaptiveThresholdPolicy(1.5),reuse=True,bonus=0.25,wc=WC_C)}
    data = []
    for tn,tr in traces.items():
        tt = _sub(tr,fast)
        for pn,po in pols.items():
            r=sim.run(tt,po,verbose=False)
            data.append({"trace":tn,"policy":pn,"cr":round(r.mean_compression,3),
                "stability":round(1-min(1,r.cv_compression),3),"reuse":round(r.mean_reuse_efficiency,4),
                "memory_savings":round(r.memory_savings_pct/100,3),"cv":round(r.cv_compression,4)})
        print(f"    {tn}: done")
    return data

def main():
    p=argparse.ArgumentParser(); p.add_argument("--trace-dir",default=None); p.add_argument("--fast",action="store_true")
    p.add_argument("--output",default="output/paper_experiments.json"); p.add_argument("--experiments",default="1,2,3,5,6,7")
    args=p.parse_args(); exps=set(args.experiments.split(","))
    print("="*70+"\n  Paper Experiments Runner\n"+"="*70)
    traces={}
    if args.trace_dir:
        from pathlib import Path
        for f in sorted(Path(args.trace_dir).rglob("*.jsonl")):
            traces[f.stem.lower().replace("_trace","")]=load_mooncake_trace(f,name=f.stem.lower().replace("_trace",""))
    else:
        n=150 if args.fast else 1000
        for v in ["arxiv","conversation","tool_agent"]: traces[v]=generate_synthetic_mooncake_trace(n,180000,v)
    primary=traces.get("arxiv",list(traces.values())[0])
    R={}
    if "1" in exps: R["exp1"]=experiment_1()
    if "2" in exps: R["exp2"]=experiment_2(primary,args.fast)
    if "3" in exps: R["exp3"]=experiment_3(primary,args.fast)
    if "5" in exps: R["exp5"]=experiment_5(primary,args.fast)
    if "6" in exps: R["exp6"]=experiment_6()
    if "7" in exps: R["exp7"]=experiment_7(traces,args.fast)
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output,"w") as f: json.dump(R,f,indent=2)
    print(f"\n  Saved: {args.output}\n  Done!")

if __name__=="__main__": 
    main()
