import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = maskrcnn_resnet50_fpn(weights="DEFAULT").to(device)
model.eval()

cap = cv2.VideoCapture("parking_space.mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

grid_size = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = cv2.getTickCount()

    frame = cv2.resize(frame, (960, 540))

    img_tensor = F.to_tensor(frame).to(device)
    img_tensor = [img_tensor]

    with torch.no_grad():
        prediction = model(img_tensor)

    occupied_spaces = []
    if len(prediction) > 0 and "masks" in prediction[0]:
        for i in range(len(prediction[0]["masks"])):
            score = prediction[0]["scores"][i].item()
            label = prediction[0]["labels"][i].item()

            if score > 0.7 and label == 3:
                mask = prediction[0]["masks"][i, 0].mul(255).byte().cpu().numpy()
                occupied_spaces.append(mask)

    height, width, _ = frame.shape
    occupied_grid = np.zeros((height // grid_size, width // grid_size), dtype=np.uint8)

    for mask in occupied_spaces:
        mask_resized = cv2.resize(mask, (width // grid_size, height // grid_size))
        occupied_grid |= (mask_resized > 128).astype(np.uint8)

    for j in range(occupied_grid.shape[0]):
        for i in range(occupied_grid.shape[1]):
            color = (0, 0, 255) if occupied_grid[j, i] else (0, 255, 0)
            cv2.rectangle(frame, (i * grid_size, j * grid_size), ((i + 1) * grid_size, (j + 1) * grid_size), color, 2)

    end_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (end_time - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Fast Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
