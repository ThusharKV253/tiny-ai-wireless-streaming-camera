# tiny_ai_cam.py
import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- Helpers for downloading images (optional) ----------
def download_from_google(keyword, limit=20, out_dir='data/google'):
    """
    Try google_images_download/simple-image-download. May fail sometimes.
    If it fails, use the bing downloader fallback (recommended).
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        # simple_image_download interface
        from simple_image_download import simple_image_download as sid
        downloader = sid.simple_image_download
        downloader().download(keyword, limit)
        # library creates ./images/<keyword> by default; move contents
        src = Path('images') / keyword
        if src.exists():
            for f in src.iterdir():
                f.rename(Path(out_dir)/f.name)
            # cleanup
            try:
                import shutil
                shutil.rmtree('images')
            except Exception:
                pass
        print("Downloaded (google) ->", out_dir)
    except Exception as e:
        print("google downloader failed:", e)
        print("Trying bing-image-downloader fallback...")
        download_from_bing(keyword, limit, out_dir)

def download_from_bing(keyword, limit=20, out_dir='data/bing'):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from bing_image_downloader import downloader
        downloader.download(keyword, limit=limit, output_dir=out_dir, adult_filter_off=True, force_replace=False, timeout=60)
        print("Downloaded (bing) ->", out_dir)
    except Exception as e:
        print("bing downloader failed:", e)
        print("Please download images manually if both fail.")

# ---------- Evaluate images in a folder ----------
def evaluate_on_folder(model, folder, save_annotated=False):
    folder = Path(folder)
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    if not images:
        print("No images found in", folder)
        return
    out_dir = folder / "results"
    out_dir.mkdir(exist_ok=True)
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = model.predict(img, conf=0.35, iou=0.45, verbose=False)
        annotated = results[0].plot()  # returns annotated image (numpy)
        if save_annotated:
            cv2.imwrite(str(out_dir / f"ann_{img_path.name}"), annotated)
        # Print top predictions
        boxes = results[0].boxes
        print(f"\n{img_path.name}: {len(boxes)} boxes")
        for b in boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            print(f"  class {cls} conf {conf:.2f}")
    print("Done. Annotated results saved to", out_dir)

# ---------- Webcam / streaming UI ----------
def run_camera(model, source=0, save_frames_dir=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {source}")
    if save_frames_dir:
        Path(save_frames_dir).mkdir(parents=True, exist_ok=True)

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame not received")
            break
        results = model.predict(frame, conf=0.35, iou=0.45, verbose=False)
        annotated = results[0].plot()
        # compute fps
        dt = time.time() - fps_time
        fps = 1/dt if dt>0 else 0.0
        fps_time = time.time()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Tiny AI Cam - press q to quit, s to save", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            if save_frames_dir:
                save_path = Path(save_frames_dir) / f"frame_{int(time.time())}.jpg"
                cv2.imwrite(str(save_path), frame)
                print("Saved", save_path)
    cap.release()
    cv2.destroyAllWindows()

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=0, help='webcam index or video file')
    parser.add_argument('--weights', default='yolov8n.pt', help='model weights (pretrained)')
    parser.add_argument('--eval_dir', default=None, help='evaluate model on image folder and exit')
    parser.add_argument('--download', default=None, help='keyword to download images from Google (optional)')
    parser.add_argument('--download_limit', type=int, default=20)
    parser.add_argument('--save_frames', default='saved_frames', help='folder to save frames when pressing s')
    args = parser.parse_args()

    # Load model
    print("Loading model:", args.weights)
    model = YOLO(args.weights)  # will download weights automatically if not present

    if args.download:
        print("Downloading images for keyword:", args.download)
        download_from_google(args.download, limit=args.download_limit, out_dir=f"data/{args.download.replace(' ','_')}")

    if args.eval_dir:
        print("Evaluating on folder:", args.eval_dir)
        evaluate_on_folder(model, Path(args.eval_dir), save_annotated=True)
        return

    # run live camera
    run_camera(model, source=int(args.source) if str(args.source).isdigit() else args.source, save_frames_dir=args.save_frames)

if __name__ == '__main__':
    main()
