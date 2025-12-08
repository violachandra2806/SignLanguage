import os

def check_training_status():
    """Check if training has started and created any files"""
    
    training_dir = 'models/trained/sibi_best_v1'
    
    if os.path.exists(training_dir):
        print("✅ Training folder exists!")
        
        # List all files in training directory
        for root, dirs, files in os.walk(training_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"📁 {file_path} ({file_size} bytes)")
                
        # Check for weights specifically
        weights_dir = os.path.join(training_dir, 'weights')
        if os.path.exists(weights_dir):
            print("✅ Weights folder exists!")
        else:
            print("⏳ Weights folder not created yet (still in first epoch)")
            
    else:
        print("⏳ Training folder doesn't exist yet (still in first epoch)")

if __name__ == "__main__":
    check_training_status()