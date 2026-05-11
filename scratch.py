import cv2
import numpy as np

# Test optical flow shape
bbox = [100, 100, 200, 200]
x1, y1, x2, y2 = bbox
pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
print("Points:", pts.shape)
