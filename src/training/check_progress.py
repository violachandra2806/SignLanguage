import os
import sys
from ultralytics import YOLO

def check_training_progress():
    """Check how much training has completed"""
    try:
        # ✅ FIXED: Correct path
        checkpoint_path = 'runs/detect/sibi_guaranteed/weights/last.pt'
        
        if os.path.exists(checkpoint_path):
            model = YOLO(checkpoint_path)
            
            # Get training results
            results = model.val()
            print(f"✅ Current mAP50: {results.box.map50:.3f}")
            print(f"📊 Current mAP50-95: {results.box.map:.3f}")
            print(f"📈 Training is resumable from last.pt")
            print(f"🔍 Checkpoint exists at: {checkpoint_path}")
        else:
            print(f"❌ No checkpoint found at: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Error checking progress: {e}")

if __name__ == "__main__":
    check_training_progress()