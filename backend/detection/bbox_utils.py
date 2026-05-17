import cv2
import numpy as np
import torch

def get_bboxes_from_mask(mask, threshold=0.5):
    """
    Derive bounding boxes from a segmentation mask.
    Returns a list of [x, y, w, h] coordinates.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # Convert to binary mask
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter small boxes (noise)
        if w > 5 and h > 5:
            bboxes.append([int(x), int(y), int(w), int(h)])
            
    return bboxes

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.
    """
    img_copy = image.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(img_copy, "Abnormality", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy
