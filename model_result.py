import os
from ultralytics import YOLO

def evaluate_model(model_path):
    """
    Evaluate a trained YOLOv8 model and display accuracy (mAP50 & mAP50-95)
    """

    # Load the trained model
    model = YOLO(model_path)

    print("🧪 Running validation on model...")
    metrics = model.val()

    # Extract metrics
    mAP50 = metrics.box.map50          # Accuracy (@ IoU=0.5)
    mAP5095 = metrics.box.map          # Strict accuracy (IoU 0.5-0.95)

    # Print results
    print("\n📌 Model Evaluation Result")
    print("---------------------------------")
    print(f"📊 Accuracy (mAP50): {mAP50:.3f}  ({mAP50*100:.2f}%)")
    print(f"📊 Strict Accuracy (mAP50-95): {mAP5095:.3f}  ({mAP5095*100:.2f}%)")
    print("---------------------------------\n")

    return mAP50, mAP5095


if __name__ == "__main__":
    # Change this to your actual weights result
    model_weights = "runs/detect/sibi_guaranteed/weights/best.pt"
    
    if not os.path.exists(model_weights):
        print(f"❌ ERROR: Model weights not found at: {model_weights}")
    else:
        evaluate_model(model_weights)
