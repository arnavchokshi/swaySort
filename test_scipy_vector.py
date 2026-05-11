import time
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

def _smooth_boxes_cpu(boxes, medfilt_window, sigma):
    smoothed = np.empty_like(boxes)
    for c in range(boxes.shape[1]):
        s = medfilt(boxes[:, c], kernel_size=medfilt_window)
        s = gaussian_filter1d(s, sigma=sigma)
        smoothed[:, c] = s
    return smoothed

def _smooth_boxes_vector(boxes, medfilt_window, sigma):
    smoothed = medfilt(boxes, kernel_size=(medfilt_window, 1))
    smoothed = gaussian_filter1d(smoothed, sigma=sigma, axis=0)
    return smoothed

boxes = np.random.rand(1000, 4)
medfilt_window = 11
sigma = 3.0

t0 = time.time()
for _ in range(1000):
    cpu_res = _smooth_boxes_cpu(boxes, medfilt_window, sigma)
print("CPU Loop:", time.time() - t0)

t0 = time.time()
for _ in range(1000):
    vec_res = _smooth_boxes_vector(boxes, medfilt_window, sigma)
print("Vectorized:", time.time() - t0)

print("Max diff:", np.abs(cpu_res - vec_res).max())
