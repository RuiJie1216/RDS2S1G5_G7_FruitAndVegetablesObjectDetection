"""Detect multiple fruits and vegetables in one complete image."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from config import DEFAULT_MODEL
from run_manager import latest_model
from train import load_detector_checkpoint, resolve_device

def resolve_model_path(requested: str|Path|None=None)->Path:
    candidate=requested or DEFAULT_MODEL
    if candidate:
        path=Path(candidate).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(f"Model not found: {path}")
        return path
    return latest_model()

def load_detector(model_path: str|Path|None=None, device_name="auto"):
    path=resolve_model_path(model_path); device=resolve_device(device_name); model,checkpoint,names=load_detector_checkpoint(path,device)
    return model,path,names

def format_detections(output,class_names,confidence=.25):
    rows=[]
    for box,label,score in zip(output["boxes"].detach().cpu().tolist(),output["labels"].detach().cpu().tolist(),output["scores"].detach().cpu().tolist()):
        if score<confidence: continue
        rows.append({"class_id":int(label)-1,"class_name":class_names[int(label)-1],"confidence":float(score),"bbox":[float(v) for v in box]})
    return rows

@torch.no_grad()
def predict_image(model,class_names,image_path,confidence=.25,nms_threshold=.45,max_detections=200,output_path:Path|None=None):
    model.score_thresh=float(confidence); model.nms_thresh=float(nms_threshold); model.detections_per_img=int(max_detections)
    device=getattr(model,"_inference_device",next(model.parameters()).device)
    image=Image.open(image_path).convert("RGB"); tensor=F.convert_image_dtype(F.pil_to_tensor(image),torch.float32).to(device)
    output=model([tensor])[0]; detections=format_detections(output,class_names,confidence)
    annotated=image.copy(); draw=ImageDraw.Draw(annotated)
    for row in detections:
        box=row["bbox"]; draw.rectangle(box,outline="lime",width=4); draw.text((box[0]+2,box[1]+2),f"{row['class_name']} {row['confidence']:.2f}",fill="lime",stroke_width=2,stroke_fill="black")
    if output_path is not None:
        output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True); annotated.save(output_path)
    return detections,annotated

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("image",type=Path); p.add_argument("--model",type=Path); p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--confidence",type=float,default=.25); p.add_argument("--nms-threshold",type=float,default=.45); p.add_argument("--max-detections",type=int,default=200); p.add_argument("--output",type=Path)
    a=p.parse_args(); model,path,names=load_detector(a.model,a.device); output=a.output or a.image.with_name(a.image.stem+"_detected.jpg")
    detections,_=predict_image(model,names,a.image,a.confidence,a.nms_threshold,a.max_detections,output)
    print(json.dumps({"model":str(path),"image":str(a.image),"annotated_image":str(output),"num_detections":len(detections),"detections":detections},indent=2,ensure_ascii=False))
if __name__=="__main__": main()
