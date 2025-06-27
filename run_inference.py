from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")  # make sure this path is correct

# Set image path directly
image_path = r"C:\Users\Hp\Downloads\odataset\test\images\IMG_9573_JPG.rf.fe782eccfc92afcc231833cba85ef3d3.jpg"

# Run inference
results = model(image_path)

# Parse and print results
for result in results:
    boxes = result.boxes
    class_ids = boxes.cls
    confidences = boxes.conf
    names = result.names

    print("\nDetected objects:")
    for i, cls in enumerate(class_ids):
        label = names[int(cls)]
        confidence = confidences[i].item()
        print(f"- {label} ({confidence*100:.2f}%)")
