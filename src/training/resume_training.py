import os
import sys
from ultralytics import YOLO

def resume_training():
    """Resume training from the last checkpoint"""
    
    model = YOLO('runs/detect/sibi_guaranteed/weights/last.pt')
    
    results = model.train(
        resume=True,
        data='data.yaml',
        epochs=60,           # Updated to match guaranteed training
        imgsz=320,           # Matches guaranteed training
        batch=16,            # Matches guaranteed training
        patience=20,         # Updated to match guaranteed training
        device='cpu',
        workers=1,           # Matches guaranteed training
        lr0=0.01,
        save=True,
        exist_ok=True,
        name='sibi_guaranteed',  # Updated name
        cos_lr=True,
        optimizer='Adam',    # Matches guaranteed training
        fliplr=0.3,          # Matches guaranteed training
    )
    
    print("✅ Training resumed and completed!")
    metrics = model.val()
    print(f"📊 Final mAP50: {metrics.box.map50:.3f}")
    
    return results, metrics

if __name__ == "__main__":
    resume_training()