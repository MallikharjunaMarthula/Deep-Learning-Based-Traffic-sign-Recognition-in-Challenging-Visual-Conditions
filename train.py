from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 nano model (recommended for CPU)
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="data.yaml",          # Path to dataset config
        epochs=30,                 # Number of training epochs
        imgsz=640,                 # Image size
        batch=8,                   # Batch size (reduce if system slow)
        name="traffic_signs_kaggle",  # Folder name inside runs/detect
        device="cpu",              # Use "0" if you have NVIDIA GPU
        verbose=True
    )

if __name__ == "__main__":
    main()