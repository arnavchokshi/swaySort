import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

def _smooth_boxes_cpu(boxes, medfilt_window, sigma):
    smoothed = np.empty_like(boxes)
    for c in range(boxes.shape[1]):
        s = medfilt(boxes[:, c], kernel_size=medfilt_window)
        s = gaussian_filter1d(s, sigma=sigma)
        smoothed[:, c] = s
    return smoothed

def _smooth_boxes_gpu(boxes_np, medfilt_window, sigma, device='mps'):
    # Move to tensor (C, N) where C=4
    # boxes_np: (N, 4) -> (1, 4, N)
    x = torch.from_numpy(boxes_np).float().T.unsqueeze(0).to(device)
    
    # Pad for median filter
    pad_m = medfilt_window // 2
    x_padded = F.pad(x, (pad_m, pad_m), mode='reflect') # (1, 4, N+2pad)
    
    # unfold along last dimension (N), window size, stride 1
    # shape: (1, 4, N, window)
    unfolded = x_padded.unfold(2, medfilt_window, 1)
    
    # calc median along last dim
    x = unfolded.median(dim=-1).values # (1, 4, N)
    
    # Gaussian kernel
    radius = int(4 * sigma + 0.5)
    size = 2 * radius + 1
    x_grid = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x_grid / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size).repeat(4, 1, 1) # groups=4 so each channel independent
    
    pad_g = radius
    x_padded = F.pad(x, (pad_g, pad_g), mode='reflect')
    x = F.conv1d(x_padded, kernel, groups=4)
    
    return x.squeeze(0).T.cpu().numpy()

boxes = np.random.rand(1000, 4)
medfilt_window = 11
sigma = 3.0

t0 = time.time()
for _ in range(100):
    cpu_res = _smooth_boxes_cpu(boxes, medfilt_window, sigma)
print("CPU:", time.time() - t0)

# Warmup
gpu_res = _smooth_boxes_gpu(boxes, medfilt_window, sigma)

t0 = time.time()
for _ in range(100):
    gpu_res = _smooth_boxes_gpu(boxes, medfilt_window, sigma)
print("GPU:", time.time() - t0)

print("Max diff:", np.abs(cpu_res - gpu_res).max())
