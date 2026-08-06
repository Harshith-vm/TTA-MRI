#!/usr/bin/env python3
"""Post-process results into corrected tables + publication figures.
Run AFTER the GPU re-run. Reads <ROOT>/results/all_results.jsonl and writes to
<ROOT>/figures/output_v2/ (figures) and .../tables/ (CSVs).
ROOT = env MRI_ROOT, else this script's parent-of-parent (repo root)."""
import os, sys, json, glob, csv
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.environ.get("MRI_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "all_results.jsonl")
SALDIR = os.path.join(ROOT, "results", "saliency")
OUT = os.path.join(ROOT, "figures", "output_v2"); os.makedirs(OUT, exist_ok=True)
TAB = os.path.join(OUT, "tables"); os.makedirs(TAB, exist_ok=True)

MODELS = ["resnet50", "efficientnet_b3", "swin_t", "vit_b16", "phikon_v2"]
ML = {"resnet50":"ResNet-50","efficientnet_b3":"EfficientNet-B3","swin_t":"Swin-T",
      "vit_b16":"ViT-B/16","phikon_v2":"Phikon-v2"}
MLAB = {"no_tta":"No-TTA","bn_adapt":"BN-Adapt","tent":"Tent","eata":"EATA","sar":"SAR",
    "cotta":"CoTTA","rotta":"RoTTA","t3a":"T3A","deyo":"DeYO","lame":"LAME","roid":"ROID",
    "sotta":"SoTTA","vida":"ViDA","come":"COME","rem":"REM","sicl":"SICL","rmemsafe":"RMemSafe"}
FAM = {"no_tta":"Baseline","bn_adapt":"Norm","t3a":"Prototype","lame":"Grad-free","sicl":"Grad-free",
    "tent":"Entropy","eata":"Entropy","roid":"Entropy","come":"Entropy","rem":"Entropy",
    "sar":"Sharpness","sotta":"Sharpness","deyo":"Entropy","vida":"Adapter",
    "rmemsafe":"Continual","rotta":"Continual","cotta":"Continual"}
FC = {"Baseline":"#000","Norm":"#999","Prototype":"#56B4E9","Grad-free":"#009E73",
    "Entropy":"#0072B2","Sharpness":"#E69F00","Adapter":"#CC79A7","Continual":"#D55E00"}
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
    "axes.spines.right":False,"savefig.dpi":400,"pdf.fonttype":42})

rows = []
with open(SRC) as f:
    for line in f:
        try:
            r = json.loads(line)
            if r["model"] in MODELS: rows.append(r)
        except Exception: pass
print(f"loaded {len(rows)} rows")
def esi(r): return r["explanation_stability"]["mean_esi"]
methods = sorted({r["tta_method"] for r in rows},
    key=lambda m:-np.mean([esi(r) for r in rows if r["tta_method"]==m and esi(r) is not None]))
def m_(sel,k):
    v=[k(r) for r in sel if k(r) is not None]; return (np.mean(v),len(v)) if v else (np.nan,0)
def boot(x,n=2000):
    x=np.asarray([v for v in x if v is not None]); rng=np.random.default_rng(0)
    b=[rng.choice(x,len(x),True).mean() for _ in range(n)]
    return x.mean(),np.percentile(b,2.5),np.percentile(b,97.5)

# ---------- TABLES ----------
with open(f"{TAB}/table1_summary.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["method","pre_AUC","post_AUC","dAUC","pre_ECE","post_ECE","dECE","ESI","collapse","n"])
    for m in methods:
        s=[r for r in rows if r["tta_method"]==m]
        preA=m_(s,lambda r:r["pre_tta"]["auc"])[0]; postA=m_(s,lambda r:r["post_tta"]["auc"])[0]
        preE=m_(s,lambda r:r["pre_tta"]["ece"])[0]; postE=m_(s,lambda r:r["post_tta"]["ece"])[0]
        dA=m_(s,lambda r:r["deltas"]["delta_auc"])[0]; dE=m_(s,lambda r:r["deltas"]["delta_ece"])[0]
        e,n=m_(s,esi)
        w.writerow([MLAB[m],f"{preA:.4f}",f"{postA:.4f}",f"{dA:+.4f}",f"{preE:.4f}",
                    f"{postE:.4f}",f"{dE:+.4f}",f"{e:.4f}","YES" if postA<0.7 else "",n])
with open(f"{TAB}/table3_esi_threshold.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["method","frac<0.98","frac<0.95","frac<0.90","n"])
    for m in methods:
        e=np.array([esi(r) for r in rows if r["tta_method"]==m and esi(r) is not None])
        w.writerow([MLAB[m],f"{np.mean(e<.98):.3f}",f"{np.mean(e<.95):.3f}",f"{np.mean(e<.90):.3f}",len(e)])
def kl(r):
    px=r["explanation_stability"]["per_xai"].get("integrated_gradients") or \
       r["explanation_stability"]["per_xai"].get("gradcam")
    return px.get("mean_kl_div") if px else None
with open(f"{TAB}/table4_kl_vs_perf.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["scope","rho(KL,|dAUC|)","p","rho(KL,dECE)","p","n"])
    for scope,sel in [("ALL",rows)]+[(MLAB[m],[r for r in rows if r["tta_method"]==m]) for m in methods]:
        K=[];A=[];E=[]
        for r in sel:
            k=kl(r)
            if k is None: continue
            K.append(k); A.append(abs(r["deltas"]["delta_auc"])); E.append(r["deltas"]["delta_ece"])
        if len(K)>10 and np.std(K)>0:
            ra,pa=spearmanr(K,A); re,pe=spearmanr(K,E)
            w.writerow([scope,f"{ra:.3f}",f"{pa:.1e}",f"{re:.3f}",f"{pe:.1e}",len(K)])
with open(f"{TAB}/table5_silent_failures.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["definition","count","of","top_methods"])
    for name,fn in [
        ("stored flag (|dAUC|<.02 & dECE>.05)", lambda r:r["silent_failure"]),
        ("prediction collapse (post_AUC<.70)", lambda r:r["post_tta"]["auc"]<0.70)]:
        hits=[r for r in rows if fn(r)]; bym=defaultdict(int)
        for r in hits: bym[MLAB[r["tta_method"]]]+=1
        top=", ".join(f"{k}({v})" for k,v in sorted(bym.items(),key=lambda x:-x[1])[:4])
        w.writerow([name,len(hits),len(rows),top])
print("tables ->", TAB)

def save(fig,n):
    for e in ("pdf","png"): fig.savefig(f"{OUT}/{n}.{e}",bbox_inches="tight",facecolor="white")
    plt.close(fig)

# fig1 heatmap
grid=np.full((len(MODELS),len(methods)),np.nan); cell=defaultdict(list)
for r in rows:
    if esi(r) is not None: cell[(r["model"],r["tta_method"])].append(esi(r))
for i,md in enumerate(MODELS):
    for j,mt in enumerate(methods):
        if cell.get((md,mt)): grid[i,j]=np.mean(cell[(md,mt)])
fig,ax=plt.subplots(figsize=(7.2,2.6)); im=ax.imshow(grid,aspect="auto",cmap="RdYlGn",vmin=0.85,vmax=1.0)
ax.set_xticks(range(len(methods))); ax.set_xticklabels([MLAB[m] for m in methods],rotation=55,ha="right")
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([ML[m] for m in MODELS])
for i in range(len(MODELS)):
    for j in range(len(methods)):
        if not np.isnan(grid[i,j]): ax.text(j,i,f"{grid[i,j]:.2f}",ha="center",va="center",fontsize=5,color="white" if grid[i,j]<0.93 else "black")
fig.colorbar(im,ax=ax,fraction=0.025,pad=0.01).set_label("Mean ESI",fontsize=7)
ax.set_title("Explanation stability (ESI) across models and TTA methods"); save(fig,"fig1_esi_heatmap")

# fig2 ranking forest
st=sorted([(m,*boot([esi(r) for r in rows if r["tta_method"]==m])) for m in methods],key=lambda t:t[1])
fig,ax=plt.subplots(figsize=(4.0,4.2)); ys=np.arange(len(st))
for y,(m,me,lo,hi) in zip(ys,st):
    ax.plot([lo,hi],[y,y],color=FC[FAM[m]],lw=1.4); ax.scatter([me],[y],color=FC[FAM[m]],s=22,edgecolors="k",lw=.4,zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([MLAB[m] for m,*_ in st]); ax.set_xlabel("Mean ESI (95% CI)")
ax.legend(handles=[Patch(color=FC[f],label=f) for f in ["Continual","Sharpness","Entropy","Adapter","Norm","Prototype","Grad-free","Baseline"]],loc="upper left",frameon=False)
ax.set_title("TTA methods ranked by explanation disruption"); ax.grid(axis="x",alpha=.3,lw=.5); save(fig,"fig2_ranking")

# fig9 saliency-by-method (bn_adapt/sar/cotta) for a representative run
def find(m):
    c=glob.glob(f"{SALDIR}/camelyon17/*/{m}/d0_s17/qualitative.npz") or glob.glob(f"{SALDIR}/**/{m}/**/qualitative.npz",recursive=True)
    return c[0] if c else None
panel=[(x,MLAB[x]) for x in ["bn_adapt","sar","cotta"] if find(x)]
if panel:
    fig,axes=plt.subplots(len(panel),3,figsize=(3.7,1.25*len(panel)+.3),squeeze=False)
    xai="integrated_gradients"
    for r_i,(m,lab) in enumerate(panel):
        npz=np.load(find(m)); key=f"{xai}_before"
        if key not in npz: xai="gradcam"; key=f"{xai}_before"
        img=npz["img"][0].transpose(1,2,0); img=(img-img.min())/(np.ptp(img)+1e-8)
        axes[r_i][0].imshow(img); axes[r_i][0].set_ylabel(lab,fontsize=8,fontweight="bold")
        axes[r_i][1].imshow(npz[f"{xai}_before"][0],cmap="jet"); axes[r_i][2].imshow(npz[f"{xai}_after"][0],cmap="jet")
        for c_i,t in zip(range(3),["H&E","pre-TTA","post-TTA"]):
            axes[r_i][c_i].set_xticks([]); axes[r_i][c_i].set_yticks([])
            if r_i==0: axes[r_i][c_i].set_title(t,fontsize=8)
    fig.suptitle("Saliency: pre vs post-TTA",fontsize=8); save(fig,"fig9_saliency_by_method")
print("figures ->", OUT)
