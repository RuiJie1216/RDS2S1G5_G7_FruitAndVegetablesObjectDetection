"""Evaluate full-image detections on an explicit validation or test split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from config import DATA_ROOT
from detection_data import YoloDetectionDataset, collate_detection_batch
from detection_metrics import collect_predictions, compute_detection_metrics, inference_efficiency, save_evaluation_artifacts
from train import load_detector_checkpoint, make_loader, resolve_device
from run_manager import record_result

def evaluate_checkpoint(model_path:Path,split:str,device_name:str,dataset_root:Path,batch_size:int,workers:int,confidence:float,ap_score_threshold:float=.001,max_images:int=0):
    device=resolve_device(device_name); model,checkpoint,names=load_detector_checkpoint(model_path,device)
    dataset=YoloDetectionDataset(dataset_root,split,False); loader=make_loader(dataset,batch_size,False,workers,device)
    records=collect_predictions(model,loader,device,ap_score_threshold,max_images); metrics,per_class,confusion=compute_detection_metrics(records,63,.5,confidence)
    sample,_=dataset[0]; efficiency=inference_efficiency(model,sample,device,1 if max_images else 3,2 if max_images else 10)
    total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics.update(efficiency); metrics.update({"total_parameters":total,"trainable_parameters":trainable,"checkpoint_size_bytes":Path(model_path).stat().st_size})
    metadata={"task_type":"object_detection","evaluation_split":split,"num_classes":63,"iou_threshold":.5,"confidence_threshold":confidence,"ap_score_threshold":ap_score_threshold,"input_resolution":checkpoint["input_image_size"],"checkpoint":str(Path(model_path).resolve()),"device":str(device),"test_set_accessed":split=="test"}
    payload=save_evaluation_artifacts(Path(model_path).resolve().parent,split,records,metrics,per_class,confusion,names,metadata); record_result(payload); return payload

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",type=Path,required=True); p.add_argument("--split",choices=("val","test"),required=True); p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto"); p.add_argument("--dataset",type=Path,default=DATA_ROOT); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--confidence",type=float,default=.25); p.add_argument("--ap-score-threshold",type=float,default=.001); p.add_argument("--max-images",type=int,default=0)
    a=p.parse_args(); print(json.dumps(evaluate_checkpoint(a.model,a.split,a.device,a.dataset,a.batch_size,a.num_workers,a.confidence,a.ap_score_threshold,a.max_images),indent=2))
if __name__=="__main__": main()
