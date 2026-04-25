import time
import subprocess
import os
import sys
import joblib
import cv2
import numpy as np

def run_pipeline(video_path, out_path, adaptive_stride):
    env = os.environ.copy()
    env["BEST_ID_ADAPTIVE_ANCHOR"] = str(adaptive_stride)
    
    start = time.time()
    cmd = [
        sys.executable, "-m", "tracking.run_pipeline",
        "--video", video_path,
        "--out", out_path,
        "--device", "mps",
        "--force"
    ]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time()
    return end - start

def load_tracks(pkl_path):
    # Returns dict[track_id -> Track]
    return joblib.load(pkl_path)

def draw_boxes(frame, frame_idx, tracks, color, title):
    cv2.putText(frame, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    for tid, tr in tracks.items():
        frames = tr.frames
        if frame_idx in frames:
            pos = np.where(frames == frame_idx)[0][0]
            box = tr.bboxes[pos]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def render_side_by_side(video_path, tracks_base, tracks_adapt, out_vid_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_vid_path, fourcc, fps, (w*2, h))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_base = frame.copy()
        frame_adapt = frame.copy()
        
        frame_base = draw_boxes(frame_base, frame_idx, tracks_base, (0, 0, 255), "Baseline (YOLO+ReID every frame)")
        frame_adapt = draw_boxes(frame_adapt, frame_idx, tracks_adapt, (0, 255, 0), "Adaptive (LK Flow propagate)")
        
        combined = np.hstack((frame_base, frame_adapt))
        out.write(combined)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Saved render to {out_vid_path}")


def main():
    videos = [
        "/Users/arnavchokshi/Desktop/CV_pipeline/CVAT/gymTest/IMG_8309.mov",
        "/Users/arnavchokshi/Desktop/CV_pipeline/CVAT/darkTest/darkTest.mov"
    ]
    
    stride = 4 # Anchor every 4 frames
    
    for vid in videos:
        name = os.path.basename(vid).split('.')[0]
        print(f"\n--- Processing {name} ---")
        
        base_out = f"work/demos/{name}_base_tracks.pkl"
        adapt_out = f"work/demos/{name}_adapt_tracks.pkl"
        
        print("Running Baseline Pipeline...")
        t_base = run_pipeline(vid, base_out, 1)
        print(f"Baseline Time: {t_base:.2f}s")
        
        print(f"Running Adaptive Pipeline (stride={stride})...")
        t_adapt = run_pipeline(vid, adapt_out, stride)
        print(f"Adaptive Time: {t_adapt:.2f}s")
        print(f"SPEEDUP: {t_base / t_adapt:.2f}x")
        
        print("Loading tracks and rendering comparative video...")
        tr_base = load_tracks(base_out)
        tr_adapt = load_tracks(adapt_out)
        
        render_path = f"work/demos/{name}_side_by_side.mp4"
        render_side_by_side(vid, tr_base, tr_adapt, render_path)
        
if __name__ == "__main__":
    main()
