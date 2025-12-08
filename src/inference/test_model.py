from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time

class SIBIDetector:
    def __init__(self, model_path='runs/detect/sibi_guaranteed/weights/best.pt'):
        self.model = YOLO(model_path)
        self.class_names = ['Ayah', 'Halo', 'Kakak', 'Minum', 'Rumah', 'Sama-sama', 'Sehat', 'Teman', 'Terima kasih', 'Tidur', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    
    def predict_image(self, image_path, conf_threshold=0.5, save_result=False):
        """Predict on a single image with visualization"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return []
        
        results = self.model(img, conf=conf_threshold)
        predictions = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = self.class_names[class_id]
                    bbox = box.xyxy[0].tolist()
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(img, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_result:
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, img)
            print(f"✅ Saved result to: {output_path}")
        
        return predictions, img
    
    def predict_frame(self, frame, conf_threshold=0.5):
        """Predict on a single frame (for webcam/video)"""
        results = self.model(frame, conf=conf_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = self.class_names[class_id]
                    bbox = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    
                    # Draw bounding box and label
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Background for text for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(
                        f"{class_name}: {confidence:.2f}", 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                 (x1 + text_width, y1), (0, 255, 0), -1)
                    
                    # Put text
                    cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                               (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame, detections
    
    def predict_video(self, video_path, output_path='output_video.mp4'):
        """Predict on a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_with_detections, _ = self.predict_frame(frame)
            out.write(frame_with_detections)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"✅ Video saved to: {output_path}")

def test_webcam_realtime():
    """Real-time webcam testing with FPS display"""
    detector = SIBIDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("🎥 Starting real-time SIBI detection...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'c' to change confidence threshold (current: 0.5)")
    
    confidence_threshold = 0.5
    frame_count = 0
    fps = 0
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        frame_count += 1
        
        # Flip frame horizontally for mirror effect (optional)
        frame = cv2.flip(frame, 1)
        
        # Perform detection
        start_time = time.time()
        frame_with_detections, detections = detector.predict_frame(frame, conf_threshold=confidence_threshold)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Display FPS and info
        cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_detections, f"Confidence: {confidence_threshold}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_detections, f"Inference: {inference_time:.1f}ms", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_detections, f"Detections: {len(detections)}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show detected classes
        y_offset = 150
        for det in detections[:5]:  # Show first 5 detections
            text = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(frame_with_detections, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        
        # Display the frame
        cv2.imshow('SIBI Sign Language Detection - Real-time', frame_with_detections)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame_with_detections)
            print(f"📸 Saved screenshot as: {filename}")
        elif key == ord('c'):  # Change confidence threshold
            try:
                new_conf = float(input("Enter new confidence threshold (0.1-0.9): "))
                if 0.1 <= new_conf <= 0.9:
                    confidence_threshold = new_conf
                    print(f"Confidence threshold changed to: {confidence_threshold}")
                else:
                    print("Please enter a value between 0.1 and 0.9")
            except:
                print("Invalid input. Keeping current threshold.")
        elif key == ord('+'):  # Increase confidence with '+' key
            confidence_threshold = min(0.9, confidence_threshold + 0.05)
            print(f"Confidence threshold increased to: {confidence_threshold}")
        elif key == ord('-'):  # Decrease confidence with '-' key
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"Confidence threshold decreased to: {confidence_threshold}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Session Summary:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")

def test_trained_model():
    """Test the trained model comprehensively"""
    detector = SIBIDetector()
    
    test_folder = "data/raw/test/images/"
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        return
    
    sample_images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not sample_images:
        print("❌ No images found in test folder")
        return
    
    print(f"📊 Found {len(sample_images)} test images")
    
    for i, img_name in enumerate(sample_images[:3]):
        print(f"\n🔍 Testing image {i+1}/{min(3, len(sample_images))}: {img_name}")
        img_path = os.path.join(test_folder, img_name)
        
        predictions, annotated_img = detector.predict_image(img_path, save_result=True)
        
        if predictions:
            print(f"✅ Detected {len(predictions)} objects:")
            for pred in predictions:
                print(f"   - {pred['class']}: {pred['confidence']:.2f}")
            
            img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(img_rgb)
            plt.title(f"Detection Results: {img_name}")
            plt.axis('off')
            plt.show()
        else:
            print("❌ No objects detected")

if __name__ == "__main__":
    print("🧪 SIBI Sign Language Detection System")
    print("=" * 40)
    
    while True:
        print("\nSelect testing mode:")
        print("1. 📸 Test on images from test folder")
        print("2. 🎥 Real-time webcam detection")
        print("3. 📹 Test on video file")
        print("4. 📊 Evaluate model metrics")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            test_trained_model()
        elif choice == '2':
            test_webcam_realtime()
        elif choice == '3':
            detector = SIBIDetector()
            video_file = input("Enter video file path (or press Enter for default 'test_video.mp4'): ").strip()
            if not video_file:
                video_file = "test_video.mp4"
            detector.predict_video(video_file)
        elif choice == '4':
            print("Running model evaluation...")
            # Uncomment this if you have validation data
            # evaluate_model()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")