# scripts/batch_process_videos.py

import argparse
import os
import glob
from tqdm import tqdm
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from multiprocessing import Pool

# Initialize MTCNN per process
def init_mtcnn():
    global mtcnn
    mtcnn = MTCNN(keep_all=True)

def process_one_video(video_path, out_dir, frames_to_sample=8, size=(224,224), select_largest=True):
    try:
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, f"cannot_open:{video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            return 0, f"no_frames:{video_path}"

        step = max(1, total_frames // frames_to_sample)
        saved = 0
        idx = 0
        sample_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                boxes, probs = mtcnn.detect(pil)
                if boxes is not None:
                    if select_largest and len(boxes) > 1:
                        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
                        box = boxes[int(np.argmax(areas))]
                    else:
                        box = boxes[0]

                    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                    face = pil.crop((x1, y1, x2, y2)).resize(size)
                    fname = os.path.join(out_dir, f"frame_{sample_idx:06d}.jpg")
                    face.save(fname, quality=95)
                    saved += 1
                    sample_idx += 1

                if sample_idx >= frames_to_sample:
                    break
            idx += 1

        cap.release()
        return saved, None

    except Exception as e:
        return 0, f"err:{str(e)}"

def worker_wrapper(args):
    return process_one_video(*args)

def collect_video_list(src_root, subfolder):
    extensions = (".mp4", ".avi", ".mov", ".mkv")
    root = os.path.join(src_root, subfolder)
    files = []
    for r, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(extensions):
                files.append(os.path.join(r, f))
    return sorted(files)

def main(args):
    src_root = os.path.abspath(args.src_root)
    out_root = os.path.abspath(args.out_root)

    groups = [("original", "real"), ("Deepfakes", "fake")]

    tasks = []

    for src_sub, out_sub in groups:
        vids = collect_video_list(src_root, src_sub)
        print(f"Found {len(vids)} videos in {src_sub}")

        if args.subset > 0:
            vids = vids[:args.subset]

        for vpath in vids:
            base = os.path.splitext(os.path.basename(vpath))[0]
            out_dir = os.path.join(out_root, out_sub, base)

            if os.path.isdir(out_dir):
                existing = [x for x in os.listdir(out_dir) if x.endswith(".jpg")]
                if len(existing) >= args.frames_to_sample:
                    continue

            tasks.append((vpath, out_dir, args.frames_to_sample, (args.width, args.height), True))

    print(f"Total videos to process: {len(tasks)}")

    if args.workers <= 1:
        init_mtcnn()
        results = []
        for t in tqdm(tasks):
            r = worker_wrapper(t)
            results.append((t[0], r))
    else:
        with Pool(processes=args.workers, initializer=init_mtcnn) as pool:
            results = list(tqdm(pool.imap(worker_wrapper, tasks), total=len(tasks)))

    print("Processing finished.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--frames_to_sample", type=int, default=8)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--height", type=int, default=224)
    args = p.parse_args()
    main(args)
