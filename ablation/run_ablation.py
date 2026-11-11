#!/usr/bin/env python3
"""Train and evaluate the complete controlled ablation study."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from ablation_data import AblationDataset, build_variant_input
from config import BANDS, DATA, RESULTS, SEEDS, TEST, VARIANTS, Variant
from models import SingleChannelCNN, SoftMaskCNN

def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def loss(pred1, pred2, y, weight):
    supervised = .5 * (F.mse_loss(pred1, y) + F.mse_loss(pred2, y))
    consistency = F.mse_loss(pred1, pred2)
    return supervised + weight * consistency, supervised, consistency

def model_for(variant: Variant):
    return (SingleChannelCNN if variant.channels == 1 else SoftMaskCNN)()

def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one(variant: Variant, seed: int, epochs: int, batch_size: int, patience: int) -> Path:
    output = RESULTS / variant.key / f"seed_{seed}"
    checkpoint = output / "best_model.pth"
    completion = output / "manifest.json"
    # A checkpoint alone is not a completed run: it is overwritten whenever
    # validation improves.  Only the final manifest marks a valid run.
    if checkpoint.exists() and completion.exists():
        print(f"SKIP completed {checkpoint}", flush=True); return checkpoint
    if output.exists():
        for stale in output.iterdir():
            stale.unlink()
    output.mkdir(parents=True, exist_ok=True); set_seed(seed); dev = device()
    train = AblationDataset(variant, "train", seed); valid = AblationDataset(variant, "validation", seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model_for(variant).to(dev); optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history=[]; best=float("inf"); waiting=0
    for epoch in range(epochs):
        model.train(); train.set_epoch(epoch); sums=np.zeros(3); seen=0
        for x1,x2,y in train_loader:
            x1,x2,y=x1.to(dev),x2.to(dev),y.to(dev); optimizer.zero_grad()
            total,sup,con=loss(model(x1),model(x2),y,variant.consistency_weight); total.backward(); optimizer.step()
            n=len(y); sums += n*np.asarray([total.item(),sup.item(),con.item()]); seen += n
        model.eval(); vsums=np.zeros(3); vseen=0
        with torch.no_grad():
            for x1,x2,y in valid_loader:
                x1,x2,y=x1.to(dev),x2.to(dev),y.to(dev); total,sup,con=loss(model(x1),model(x2),y,variant.consistency_weight)
                n=len(y); vsums += n*np.asarray([total.item(),sup.item(),con.item()]); vseen += n
        scheduler.step(); row={"epoch":epoch+1,"train_loss":float(sums[0]/seen),"val_loss":float(vsums[0]/vseen),"train_supervised":float(sums[1]/seen),"val_supervised":float(vsums[1]/vseen),"train_consistency":float(sums[2]/seen),"val_consistency":float(vsums[2]/vseen)}; history.append(row)
        if row["val_loss"] < best:
            best=row["val_loss"]; waiting=0; torch.save(model.state_dict(),checkpoint)
        else: waiting += 1
        print(variant.key, seed, epoch+1, f"val={row['val_loss']:.6f}", flush=True)
        if waiting >= patience: break
    (output/"history.json").write_text(json.dumps(history,indent=2))
    (output/"manifest.json").write_text(json.dumps({"variant":variant.__dict__,"seed":seed,"device":str(dev),"parameters":parameter_count(model),"epochs_completed":len(history),"best_val_loss":best},indent=2))
    return checkpoint

def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    error=pred-truth; out={"macro_mae":float(np.mean(np.abs(error))),"macro_rmse":float(np.sqrt(np.mean(error**2)))}
    for i,b in enumerate(BANDS):
        out[f"{b}_mae"]=float(np.mean(np.abs(error[:,i]))); out[f"{b}_rmse"]=float(np.sqrt(np.mean(error[:,i]**2)))
        den=float(np.sum((truth[:,i]-truth[:,i].mean())**2)); out[f"{b}_r2"]=float(1-np.sum(error[:,i]**2)/den) if den else float("nan")
    return out

def predict_dataset(model, dataset, dev):
    loader=DataLoader(dataset,batch_size=64,shuffle=False); truth=[]; pred=[]
    model.eval()
    with torch.no_grad():
        for x1,x2,y in loader:
            output=.5*(model(x1.to(dev))+model(x2.to(dev))); pred.append(output.cpu().numpy()); truth.append(y.numpy())
    return np.concatenate(truth),np.concatenate(pred)

def _load_prediction_module():
    path=TEST/"predict_real_experiment.py"; spec=importlib.util.spec_from_file_location("real_preprocess",path); module=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module); return module

def real_images():
    p=_load_prediction_module(); band=DATA/"BandMapping_data"; t1=band/"20260511_fix_t4_3us_scan_t1"; t5=band/"20260511_fix_t4_3us_scan_t5"; bg=t1/"BG"
    from PIL import Image
    bgs=np.asarray(Image.open(bg/"BG_s.tif"),dtype=np.float32); bgr=np.asarray(Image.open(bg/"BG_r.tif"),dtype=np.float32)
    result={}
    for name,scan in [("scan_t1",t1),("scan_t5",t5)]:
        times=[]; images=[]
        for signal in sorted(scan.glob("*usAveS.tif"),key=p.time_value):
            reference=signal.with_name(signal.name.replace("AveS.tif","AveR.tif")); raw,fft,_=p.preprocess_pair(signal,reference,bgs,bgr); times.append(p.time_value(signal)); images.append(fft)
        result[name]=(np.asarray(times),images)
    return result

def input_real(image, variant):
    # No labels enter this construction.  The fixed geometry is the same
    # inference-time estimate used for every six-channel real-image variant.
    parameters = np.zeros(12, dtype=np.float32)
    parameters[7] = 26.0
    parameters[9] = 0.0
    return build_variant_input(
        image,
        parameters,
        variant,
        np.random.default_rng(0),
        geometry_jitter=False,
    )

def predict_real(model, variant, dev, scans):
    model.eval(); output={}
    with torch.no_grad():
        for name,(times,images) in scans.items():
            x=torch.from_numpy(np.stack([input_real(i,variant) for i in images])).to(dev); output[name]={"times":times.tolist(),"cnn":model(x).cpu().numpy().tolist()}
    return output

def evaluate_all(selected_variants: tuple[Variant, ...] = VARIANTS):
    dev=device(); benchmark={}; real_scans=real_images(); all_real={}
    for variant in selected_variants:
        for seed in SEEDS:
            path=RESULTS/variant.key/f"seed_{seed}"/"best_model.pth"
            if not path.exists(): raise FileNotFoundError(f"Missing {path}; train before evaluation.")
            model=model_for(variant).to(dev); model.load_state_dict(torch.load(path,map_location=dev)); truth,pred=predict_dataset(model,AblationDataset(variant,"validation",seed),dev)
            m=metrics(truth,pred); m.update({"variant":variant.key,"seed":seed}); benchmark[(variant.key,seed)]=m
            np.savez_compressed(RESULTS/variant.key/f"seed_{seed}"/"validation_predictions.npz",truth=truth,prediction=pred)
            all_real[(variant.key,seed)]=predict_real(model,variant,dev,real_scans)
    summarize(benchmark,all_real,selected_variants)

def summarize(rows, real, selected_variants: tuple[Variant, ...]):
    RESULTS.mkdir(exist_ok=True); values=list(rows.values()); columns=list(values[0]);
    with (RESULTS/"per_seed_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(values)
    summary=[]
    for variant in selected_variants:
        part=[r for r in values if r["variant"]==variant.key]; row={"variant":variant.key,"label":variant.label}
        for key in columns:
            if key not in {"variant","seed"}: row[key+"_mean"]=float(np.mean([x[key] for x in part])); row[key+"_std"]=float(np.std([x[key] for x in part],ddof=1))
        summary.append(row)
    real_json={f"{variant}_seed_{seed}":payload for (variant,seed),payload in real.items()}
    (RESULTS/"summary.json").write_text(json.dumps({"summary":summary,"real_image_comparison":real_json,"real_data_warning":"No independent orbital labels exist for real images; these predictions are comparisons, not real-image MAE or accuracy."},indent=2))
    with (RESULTS/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    labels=[v.key for v in selected_variants]; mean=[x["macro_mae_mean"]*100 for x in summary]; std=[x["macro_mae_std"]*100 for x in summary]
    colors=["#999999", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(labels,mean,yerr=std,capsize=5,color=colors[:len(labels)]); ax.set_ylabel("Validation macro MAE (percentage points)"); ax.set_title("Controlled ablation on the fixed synthetic benchmark"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(RESULTS/"ablation_macro_mae.png",dpi=220); plt.close(fig)
    # One fixed seed's predictions are used only for diagnostic visualizations;
    # the table above remains the three-seed mean +/- standard deviation.
    if any(variant.key == "A3" for variant in selected_variants):
        fig,axes=plt.subplots(1,5,figsize=(17,3.5))
        for i,(ax,band) in enumerate(zip(axes,BANDS)):
            arrays=np.load(RESULTS/"A3"/f"seed_{SEEDS[0]}"/"validation_predictions.npz")
            truth,pred=arrays["truth"][:,i],arrays["prediction"][:,i]
            ax.scatter(truth,pred,s=7,alpha=.45); lim=(0,max(float(truth.max()),float(pred.max()))*1.03); ax.plot(lim,lim,"k--",lw=1); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_title(band); ax.set_xlabel("True fraction")
        axes[0].set_ylabel("A3 predicted fraction"); fig.suptitle("A3 validation predictions on the fixed benchmark"); fig.tight_layout(); fig.savefig(RESULTS/"A3_prediction_truth_scatter.png",dpi=220); plt.close(fig)
        arrays=np.load(RESULTS/"A3"/f"seed_{SEEDS[0]}"/"validation_predictions.npz"); errors=arrays["prediction"]-arrays["truth"]; corr=np.corrcoef(errors.T)
        fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(5),BANDS); ax.set_yticks(range(5),BANDS); fig.colorbar(im,ax=ax,label="Error correlation"); ax.set_title("A3 orbital-error correlation"); fig.tight_layout(); fig.savefig(RESULTS/"A3_error_correlation.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for variant in selected_variants:
        history=json.loads((RESULTS/variant.key/f"seed_{SEEDS[0]}"/"history.json").read_text()); ax.plot([x["epoch"] for x in history],[x["val_loss"] for x in history],label=variant.key)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss"); ax.set_title("Validation-loss trajectories (one fixed seed)"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(RESULTS/"validation_loss_curves.png",dpi=220); plt.close(fig)
    for scan in ("scan_t1","scan_t5"):
        fig,axes=plt.subplots(1,5,figsize=(17,4),sharey=True)
        for i,(ax,band) in enumerate(zip(axes,BANDS)):
            for variant in selected_variants:
                sample=np.asarray([real[(variant.key,s)][scan]["cnn"] for s in SEEDS]); times=np.asarray(real[(variant.key,SEEDS[0])][scan]["times"]); ax.plot(times,sample[:,:,i].mean(0),marker="o",label=variant.key)
            ax.set_title(band); ax.set_xlabel("Time (μs)"); ax.grid(alpha=.25)
        axes[0].set_ylabel("Orbital fraction"); axes[-1].legend(); fig.suptitle(f"Real-image model comparison: {scan} (not ground-truth accuracy)"); fig.tight_layout(); fig.savefig(RESULTS/f"{scan}_model_comparison.png",dpi=220); plt.close(fig)
    # Conventional curve fitting is a comparison reference only.  It is not a
    # label and therefore never enters MAE/R2 calculations for real images.
    sys_path_added = str(TEST)
    import sys
    if sys_path_added not in sys.path: sys.path.insert(0, sys_path_added)
    fit_spec=importlib.util.spec_from_file_location("ablation_curvefit",TEST/"evaluate_diagnostics.py"); fit_module=importlib.util.module_from_spec(fit_spec); assert fit_spec.loader; fit_spec.loader.exec_module(fit_module)
    for scan,(times,images) in real_images().items():
        fitted=np.stack([fit_module.curve_fit_profile(image)["ratios"] for image in images])
        fig,axes=plt.subplots(1,5,figsize=(17,4),sharey=True)
        for i,(ax,band) in enumerate(zip(axes,BANDS)):
            ax.plot(times,fitted[:,i],"k--",marker="s",label="Curve fit")
            for variant in selected_variants:
                sample=np.asarray([real[(variant.key,s)][scan]["cnn"] for s in SEEDS]); ax.plot(times,sample[:,:,i].mean(0),marker="o",label=variant.key)
            ax.set_title(band); ax.set_xlabel("Time (μs)"); ax.grid(alpha=.25)
        axes[0].set_ylabel("Orbital fraction"); axes[-1].legend(fontsize=8); fig.suptitle(f"{scan}: CNN variants versus conventional five-band fit (comparison only)"); fig.tight_layout(); fig.savefig(RESULTS/f"{scan}_cnn_vs_curvefit.png",dpi=220); plt.close(fig)
    report=["# Ablation study report","","## Important boundary","Real experimental images have no independent orbital-fraction ground truth. The real-image plots are model/curve-fit comparisons, never real-image MAE or accuracy.","","## Fixed synthetic benchmark","All variants use the same 750 held-out theory examples: 250 held-out real residual, 250 PCA, and 250 mixed noise examples.","","## Results"]
    report += ["| Variant | Macro MAE (mean ± std) |","|---|---|"] + [f"| {x['variant']} | {x['macro_mae_mean']*100:.3f} ± {x['macro_mae_std']*100:.3f} percentage points |" for x in summary]
    (RESULTS/"ablation_report.md").write_text("\n".join(report)+"\n")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--all",action="store_true"); p.add_argument("--evaluate",action="store_true"); p.add_argument("--epochs",type=int,default=150); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--patience",type=int,default=20); p.add_argument("--variants",nargs="+",choices=[variant.key for variant in VARIANTS],help="Subset to train/evaluate. Use A3 A4 for the matched soft-versus-hard comparison."); args=p.parse_args()
    selected=tuple(variant for variant in VARIANTS if not args.variants or variant.key in args.variants)
    if args.all:
        for v in selected:
            for s in SEEDS: train_one(v,s,args.epochs,args.batch_size,args.patience)
        evaluate_all(selected)
    elif args.evaluate: evaluate_all(selected)
    else: p.error("use --all or --evaluate")
if __name__ == "__main__": main()
