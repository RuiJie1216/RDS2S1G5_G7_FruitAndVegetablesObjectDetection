"""Discover detector runs and export task-consistent assignment comparisons."""
from __future__ import annotations
import argparse,csv,io,json,os
from pathlib import Path
from typing import Any,Iterable
from config import COMPARISON_FILE,RESULTS_DIR,RUNS_DIR

MODEL_PATTERNS=("mobilenetv2_detector_best.pt","mobilenetv2_detector_latest.pt")
FIELDS=[("model_name","Model"),("task_type","Task"),("evaluation_split","Split"),("num_classes","Classes"),("detection_precision","Precision"),("classification_accuracy","Classification Accuracy"),("detection_recall","Recall"),("detection_f1","F1"),("mean_iou","Mean IoU"),("detection_rate_at_iou_0.5","Detection rate @0.5"),("map_50","mAP@0.5"),("map_50_95","mAP@0.5:0.95"),("total_parameters","Parameters"),("checkpoint_size_bytes","Checkpoint bytes"),("average_inference_seconds","Seconds/image"),("fps","FPS"),("input_resolution","Resolution"),("run_name","Run"),("checkpoint","Checkpoint")]

def list_runs(runs_dir:Path=RUNS_DIR):
    return sorted((p for p in Path(runs_dir).iterdir() if p.is_dir() and not p.name.startswith("_")),key=lambda p:p.stat().st_mtime,reverse=True) if Path(runs_dir).is_dir() else []
def model_path(run_dir:Path):
    return next((run_dir/name for name in MODEL_PATTERNS if (run_dir/name).is_file()),None)
def latest_model(runs_dir:Path=RUNS_DIR):
    candidate=next((model_path(run) for run in list_runs(runs_dir) if model_path(run)),None)
    if candidate is None: raise FileNotFoundError(f"No MobileNetV2 detector checkpoint under {runs_dir}; legacy crop checkpoints are ignored")
    return candidate
def load_model_results(path:Path=COMPARISON_FILE):
    if not Path(path).is_file(): return []
    try: payload=json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return []
    return [row for row in payload if isinstance(row,dict)] if isinstance(payload,list) else []
def _atomic(path:Path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(payload,indent=2),encoding="utf-8"); os.replace(temp,path)
def record_result(entry:dict[str,Any],path:Path=COMPARISON_FILE):
    rows=load_model_results(path); info=entry.get("info",{}); key=(entry.get("model_name"),info.get("run_name"),info.get("evaluation_split"))
    rows=[r for r in rows if (r.get("model_name"),r.get("info",{}).get("run_name"),r.get("info",{}).get("evaluation_split"))!=key]; rows.append(entry); _atomic(path,rows)
def _num(value): return value if isinstance(value,(int,float)) and not isinstance(value,bool) else None
def normalize(entry):
    info=entry.get("info",{}) if isinstance(entry.get("info"),dict) else {}; metrics=entry.get("metrics",{}) if isinstance(entry.get("metrics"),dict) else {}
    if info.get("is_smoke_test"): return None
    task=str(info.get("task_type") or ("object_crop_classification" if "crop" in str(entry.get("model_name","")).lower() else "object_detection"))
    row={"model_name":str(entry.get("model_name","Unknown")),"task_type":task,"evaluation_split":"val" if str(info.get("evaluation_split","val")).lower() in {"val","validation"} else str(info.get("evaluation_split"))}
    for key in ("num_classes","input_resolution","run_name","checkpoint"):
        if info.get(key) is not None: row[key]=info[key]
    if task=="object_detection":
        aliases={"detection_precision":"objectness_precision","detection_recall":"objectness_recall","detection_f1":"objectness_f1"}
        for key in ("detection_precision","classification_accuracy","detection_recall","detection_f1","mean_iou","detection_rate_at_iou_0.5","map_50","map_50_95","total_parameters","checkpoint_size_bytes","average_inference_seconds","fps"):
            value=metrics.get(key,metrics.get(aliases.get(key,"")))
            if _num(value) is not None: row[key]=value
    else: row["legacy_result"]=True
    return row
def _display(value): return "N/A" if value is None else f"{value:.4f}" if isinstance(value,float) else str(value)
def export_assignment_comparison(include_results:Iterable[Path]=(),split="val",output_dir:Path=RESULTS_DIR,primary_results:Path=COMPARISON_FILE):
    entries=[]
    for source in [Path(primary_results),*(Path(p) for p in include_results)]: entries.extend(load_model_results(source))
    rows=[row for entry in entries if (row:=normalize(entry)) is not None and row["evaluation_split"]==split]
    rows.sort(key=lambda r:(r["task_type"],r["model_name"],str(r.get("run_name",""))))
    output_dir.mkdir(parents=True,exist_ok=True); json_path=output_dir/"assignment_model_comparison.json"; csv_path=output_dir/"assignment_model_comparison.csv"; md_path=output_dir/"assignment_model_comparison.md"
    _atomic(json_path,{"evaluation_split":split,"models":rows,"note":"Only actual detection fields are compared; legacy crop-classifier rows are explicitly marked and contain no fabricated localisation metrics."})
    buf=io.StringIO(newline=""); writer=csv.DictWriter(buf,fieldnames=[label for _,label in FIELDS]); writer.writeheader(); [writer.writerow({label:_display(row.get(key)) for key,label in FIELDS}) for row in rows]; csv_path.write_text(buf.getvalue(),encoding="utf-8")
    headers=[label for _,label in FIELDS]; lines=["# Assignment Model Comparison","",f"Split: `{split}`","","| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]
    lines += ["| "+" | ".join(_display(row.get(key)).replace("|","\\|") for key,_ in FIELDS)+" |" for row in rows]; md_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"json":json_path,"csv":csv_path,"markdown":md_path}
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--include-results",type=Path,action="append",default=[]); p.add_argument("--split",choices=("val","test"),default="val"); p.add_argument("--output-dir",type=Path,default=RESULTS_DIR); a=p.parse_args()
    for name,path in export_assignment_comparison(a.include_results,a.split,a.output_dir).items(): print(f"{name}: {path}")
if __name__=="__main__": main()
