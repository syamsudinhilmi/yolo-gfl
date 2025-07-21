import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


def process_video(model, video_info, model_config, output_dir, stats):
    """Process a video with a given model and collect performance statistics"""
    cap = cv2.VideoCapture(video_info['path'])
    video_name = Path(video_info['path']).stem
    video_path = video_info['path']

    if not cap.isOpened():
        print(f"Error: Could not open video {video_info['name']}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup output directory
    model_video_dir = os.path.join(output_dir['videos'], model_config['name'])
    os.makedirs(model_video_dir, exist_ok=True)

    temp_video_path = os.path.join(model_video_dir, f"temp_{video_name}_{model_config['name']}.mp4")
    video_output_path = os.path.join(model_video_dir, f"{video_name}_{model_config['name']}.mp4")

    out = cv2.VideoWriter(temp_video_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,
                          (width, height))

    # Statistics tracking
    frame_times = []
    fire_counts = []
    smoke_counts = []
    total_frames = 0
    detections = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1

        start_time = time.time()
        results = model(frame, conf=model_config['conf_threshold'])
        inference_time = time.time() - start_time

        frame_times.append(inference_time)
        fire_count = 0
        smoke_count = 0
        frame_detections = []

        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            frame_detections.append({
                'class': cls,
                'confidence': conf,
                'bbox': box.xyxy[0].tolist()
            })

            if cls == 0:
                fire_count += 1
            elif cls == 1:
                smoke_count += 1

        fire_counts.append(fire_count)
        smoke_counts.append(smoke_count)
        detections.append(frame_detections)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Tambahkan FPS ke frame (ditampilkan pada kiri atas)
        fps_text = f"FPS: {1 / inference_time:.2f}" if inference_time > 0 else "FPS: -"
        cv2.putText(
            annotated_frame,
            fps_text,
            org=(10, height - 540),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Tampilkan frame secara langsung
        cv2.imshow(f"Inference - {model_config['name']} - {video_info['name']}", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Simpan ke video file
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Merge audio using FFmpeg
    try:
        ffmpeg_path = r"C:\Users\hilmi\ffmpeg\bin\ffmpeg.exe"
        final_temp_path = os.path.join(model_video_dir, f"final_temp_{video_name}_{model_config['name']}.mp4")

        ffmpeg_cmd = [
            ffmpeg_path,
            '-i', temp_video_path,
            '-i', video_path,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:a', 'aac',
            '-shortest',
            final_temp_path,
            '-y'
        ]

        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Successfully merged audio with annotated video for {video_info['name']}")

        if os.path.exists(video_output_path):
            os.remove(video_output_path)
        os.rename(final_temp_path, video_output_path)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    except FileNotFoundError:
        print("FFmpeg not found. Using video without audio.")
        if os.path.exists(video_output_path):
            os.remove(video_output_path)
        os.rename(temp_video_path, video_output_path)

    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}. Using video without audio.")
        if os.path.exists(video_output_path):
            os.remove(video_output_path)
        os.rename(temp_video_path, video_output_path)

    except Exception as e:
        print(f"Unexpected error: {e}. Using video without audio.")
        if os.path.exists(temp_video_path):
            if os.path.exists(video_output_path):
                os.remove(video_output_path)
            shutil.copy2(temp_video_path, video_output_path)
            os.remove(temp_video_path)

    # Store statistics
    stats[model_config['name']][video_info['name']] = {
        'avg_fps': len(frame_times) / sum(frame_times) if frame_times else 0,
        'total_fire': sum(fire_counts),
        'total_smoke': sum(smoke_counts),
        'max_fire_per_frame': max(fire_counts) if fire_counts else 0,
        'max_smoke_per_frame': max(smoke_counts) if smoke_counts else 0,
        'avg_inference': sum(frame_times) / len(frame_times) if frame_times else 0,
        'min_inference': min(frame_times) if frame_times else 0,
        'max_inference': max(frame_times) if frame_times else 0,
        'total_frames': total_frames,
        'detections': detections,
        'fire_per_frame': sum(fire_counts) / total_frames if total_frames > 0 else 0,
        'smoke_per_frame': sum(smoke_counts) / total_frames if total_frames > 0 else 0
    }




def visualize_stats(stats, output_dir, model_configs):
    """Generate comprehensive performance visualizations"""
    # Create various plots for model comparison

    # 1. FPS Comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(stats[list(stats.keys())[0]]))
    videos = list(stats[list(stats.keys())[0]].keys())

    width = 0.35
    for i, (model_name, videos_stats) in enumerate(stats.items()):
        fps_values = [v['avg_fps'] for v in videos_stats.values()]
        plt.bar(x_pos + (i * width), fps_values, width, label=model_name)

        # NEW: Add exact values on top of bars
        for j, v in enumerate(fps_values):
            plt.text(x_pos[j] + (i * width), v + 0.5, f"{v:.1f}",
                     ha='center', va='bottom', fontsize=8)

    plt.title('Average FPS by Video')
    plt.ylabel('Frames Per Second')
    plt.xticks(x_pos + width / 2, videos, rotation=45, ha='right')
    plt.legend()

    # 2. Detection Counts
    plt.subplot(2, 2, 2)
    width = 0.15
    x = np.arange(len(videos))

    for i, (model_name, videos_stats) in enumerate(stats.items()):
        fire_counts = [v['total_fire'] for v in videos_stats.values()]
        smoke_counts = [v['total_smoke'] for v in videos_stats.values()]

        # Plot bars for fire and smoke
        fire_bars = plt.bar(x + (i * width * 2), fire_counts, width, label=f'{model_name} Fire')
        smoke_bars = plt.bar(x + (i * width * 2) + width, smoke_counts, width, label=f'{model_name} Smoke')

        # NEW: Add exact count values on top of bars
        for j, v in enumerate(fire_counts):
            plt.text(x[j] + (i * width * 2), v + 20, str(v),
                     ha='center', va='bottom', fontsize=8)

        for j, v in enumerate(smoke_counts):
            plt.text(x[j] + (i * width * 2) + width, v + 20, str(v),
                     ha='center', va='bottom', fontsize=8)

    plt.title('Detection Counts by Class')
    plt.ylabel('Count')
    plt.xticks(x + width, videos, rotation=45, ha='right')
    plt.legend()

    # 3. Inference Time
    plt.subplot(2, 2, 3)

    for i, (model_name, videos_stats) in enumerate(stats.items()):
        inf_times = [v['avg_inference'] for v in videos_stats.values()]
        min_times = [v['min_inference'] for v in videos_stats.values()]
        max_times = [v['max_inference'] for v in videos_stats.values()]

        plt.errorbar(x + (i * width), inf_times,
                     yerr=[np.array(inf_times) - np.array(min_times),
                           np.array(max_times) - np.array(inf_times)],
                     fmt='o', label=model_name)

        # NEW: Add exact time values
        for j, v in enumerate(inf_times):
            plt.text(x[j] + (i * width), v + 0.01, f"{v:.3f}s",
                     ha='center', va='bottom', fontsize=8)

    plt.title('Inference Time (with min/max)')
    plt.ylabel('Seconds')
    plt.xticks(x, videos, rotation=45, ha='right')
    plt.legend()

    # 4. Model Size vs Performance
    plt.subplot(2, 2, 4)
    model_sizes = [config['size_mb'] for config in model_configs]
    model_fps = []

    for model_name in stats.keys():
        avg_fps_across_videos = np.mean([v['avg_fps'] for v in stats[model_name].values()])
        model_fps.append(avg_fps_across_videos)

    plt.scatter(model_sizes, model_fps, s=100)
    for i, model_name in enumerate(stats.keys()):
        plt.annotate(f"{model_name}\n{model_sizes[i]:.1f}MB, {model_fps[i]:.1f}FPS",
                     (model_sizes[i], model_fps[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.title('Model Size vs. Performance')
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Average FPS')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['stats'], 'performance_comparison.png'), dpi=300)

    # Additional analysis: Confidence distribution
    plt.figure(figsize=(12, 6))

    for i, (model_name, videos_stats) in enumerate(stats.items()):
        all_confidences = []
        fire_confidences = []
        smoke_confidences = []

        for video_stats in videos_stats.values():
            for frame_dets in video_stats['detections']:
                for det in frame_dets:
                    all_confidences.append(det['confidence'])
                    if det['class'] == 0:  # Fire
                        fire_confidences.append(det['confidence'])
                    elif det['class'] == 1:  # Smoke
                        smoke_confidences.append(det['confidence'])

        if all_confidences:
            plt.subplot(1, len(stats), i + 1)
            plt.hist([fire_confidences, smoke_confidences], bins=20, alpha=0.7,
                     label=['Fire', 'Smoke'], color=['red', 'gray'])
            plt.title(f'{model_name} Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['stats'], 'confidence_distribution.png'), dpi=300)

    # NEW: Detection per frame analysis
    plt.figure(figsize=(12, 8))

    for i, (model_name, videos_stats) in enumerate(stats.items()):
        fire_per_frame = [v['fire_per_frame'] for v in videos_stats.values()]
        smoke_per_frame = [v['smoke_per_frame'] for v in videos_stats.values()]

        plt.subplot(2, 1, 1)
        plt.bar(x + (i * width), fire_per_frame, width, label=f'{model_name}')

        # Add values on top of bars
        for j, v in enumerate(fire_per_frame):
            plt.text(x[j] + (i * width), v + 0.01, f"{v:.2f}",
                     ha='center', va='bottom', fontsize=8)

        plt.subplot(2, 1, 2)
        plt.bar(x + (i * width), smoke_per_frame, width, label=f'{model_name}')

        # Add values on top of bars
        for j, v in enumerate(smoke_per_frame):
            plt.text(x[j] + (i * width), v + 0.01, f"{v:.2f}",
                     ha='center', va='bottom', fontsize=8)

    plt.subplot(2, 1, 1)
    plt.title('Average Fire Detections per Frame')
    plt.ylabel('Fire Detections/Frame')
    plt.xticks(x, videos, rotation=45, ha='right')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Average Smoke Detections per Frame')
    plt.ylabel('Smoke Detections/Frame')
    plt.xticks(x, videos, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['stats'], 'detections_per_frame.png'), dpi=300)

    # NEW: Model performance radar chart
    plt.figure(figsize=(10, 10))

    # Calculate normalized metrics for radar chart
    model_metrics = {}
    for model_name in stats.keys():
        avg_fps = np.mean([v['avg_fps'] for v in stats[model_name].values()])
        avg_inference = np.mean([v['avg_inference'] for v in stats[model_name].values()])
        total_fire = sum([v['total_fire'] for v in stats[model_name].values()])
        total_smoke = sum([v['total_smoke'] for v in stats[model_name].values()])
        size_mb = next((c['size_mb'] for c in model_configs if c['name'] == model_name), 0)

        model_metrics[model_name] = {
            'FPS': avg_fps,
            'Inference Time': 1 / avg_inference,  # Invert so higher is better
            'Fire Detection': total_fire,
            'Smoke Detection': total_smoke,
            'Size Efficiency': 1 / size_mb * 100  # Invert so smaller model = higher score
        }

    # Normalize all metrics to 0-1 scale
    max_values = {}
    for metric in ['FPS', 'Inference Time', 'Fire Detection', 'Smoke Detection', 'Size Efficiency']:
        max_values[metric] = max([metrics[metric] for metrics in model_metrics.values()])

    for model_name, metrics in model_metrics.items():
        for metric in metrics:
            if max_values[metric] > 0:  # Avoid division by zero
                model_metrics[model_name][metric] /= max_values[metric]

    # Create radar chart
    categories = ['FPS', 'Inference Time', 'Fire Detection', 'Smoke Detection', 'Size Efficiency']
    N = len(categories)

    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw the chart for each model
    for model_name, metrics in model_metrics.items():
        values = [metrics[cat] for cat in categories]
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)

    plt.title('Model Performance Comparison (Normalized)', size=15)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['stats'], 'radar_comparison.png'), dpi=300)

    # NEW: Create detailed statistics table
    stats_data = {
        "Model": [],
        "Size (MB)": [],
        "Avg FPS": [],
        "Avg Inference (ms)": [],
        "Total Fire": [],
        "Total Smoke": [],
        "Fire/Frame": [],
        "Smoke/Frame": []
    }

    for model_name in stats.keys():
        size_mb = next((c['size_mb'] for c in model_configs if c['name'] == model_name), 0)
        avg_fps = np.mean([v['avg_fps'] for v in stats[model_name].values()])
        avg_inference = np.mean([v['avg_inference'] for v in stats[model_name].values()]) * 1000  # Convert to ms
        total_fire = sum([v['total_fire'] for v in stats[model_name].values()])
        total_smoke = sum([v['total_smoke'] for v in stats[model_name].values()])
        total_frames = sum([v['total_frames'] for v in stats[model_name].values()])
        fire_per_frame = total_fire / total_frames if total_frames > 0 else 0
        smoke_per_frame = total_smoke / total_frames if total_frames > 0 else 0

        stats_data["Model"].append(model_name)
        stats_data["Size (MB)"].append(f"{size_mb:.2f}")
        stats_data["Avg FPS"].append(f"{avg_fps:.2f}")
        stats_data["Avg Inference (ms)"].append(f"{avg_inference:.2f}")
        stats_data["Total Fire"].append(str(total_fire))
        stats_data["Total Smoke"].append(str(total_smoke))
        stats_data["Fire/Frame"].append(f"{fire_per_frame:.2f}")
        stats_data["Smoke/Frame"].append(f"{smoke_per_frame:.2f}")

    # Create a figure with a table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=[list(row) for row in zip(*stats_data.values())],
        colLabels=list(stats_data.keys()),
        loc='center',
        cellLoc='center'
    )

    # Adjust table styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['stats'], 'detailed_stats_table.png'), dpi=300)

    # Save raw stats to CSV for further analysis
    with open(os.path.join(output_dir['stats'], 'performance_stats.yaml'), 'w') as f:
        # Filter out the large detection arrays before saving
        filtered_stats = {}
        for model, videos in stats.items():
            filtered_stats[model] = {}
            for video, metrics in videos.items():
                filtered_metrics = {k: v for k, v in metrics.items() if k != 'detections'}
                filtered_stats[model][video] = filtered_metrics

        yaml.dump(filtered_stats, f)

    print("Visualization complete! Check the stats directory for results.")


def load_model_config(config_path):
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


from tkinter import Tk
from tkinter.filedialog import askopenfilename

def main():
    # Path konfigurasi
    CONFIG_PATH = "../config/test_cfg.yaml"

    # Load konfigurasi model dan direktori output
    config = load_model_config(CONFIG_PATH)

    # Sembunyikan jendela utama Tkinter (hanya tampilkan dialog file)
    Tk().withdraw()

    # Buka file explorer untuk memilih file video
    selected_path = askopenfilename(
        title="Pilih file video untuk inferensi",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    if not selected_path:
        print("Tidak ada file video yang dipilih. Program dihentikan.")
        return

    # Buat struktur informasi video yang diperlukan
    video_info = {
        'name': Path(selected_path).stem,
        'path': selected_path
    }

    # Buat direktori output (termasuk subfolder untuk stats per video)
    for dir_type, dir_path in config['output_dirs'].items():
        if dir_type != 'stats':  # 'stats' kita buat nanti berdasarkan nama video
            os.makedirs(dir_path, exist_ok=True)

    # Buat subdirektori output untuk statistik berdasarkan nama video
    video_stats_dir = os.path.join(config['output_dirs']['stats'], video_info['name'])

    # Overwrite: hapus isi direktori jika sudah ada
    if os.path.exists(video_stats_dir):
        shutil.rmtree(video_stats_dir)

    os.makedirs(video_stats_dir, exist_ok=True)

    # Inisialisasi statistik untuk semua model
    stats = {model['name']: {} for model in config['models']}

    # Proses semua model terhadap video terpilih
    for model_config in config['models']:
        print(f"\nProcessing Model: {model_config['name']}")

        # Hitung ukuran file model
        model_size_bytes = os.path.getsize(model_config['path'])
        model_config['size_mb'] = model_size_bytes / (1024 * 1024)

        # Load model YOLO
        model = YOLO(model_config['path'])

        print(f"  Analyzing Video: {video_info['name']}")
        process_video(
            model=model,
            video_info=video_info,
            model_config=model_config,
            output_dir={
                'videos': config['output_dirs']['videos'],
                'stats': video_stats_dir
            },
            stats=stats
        )

    # Buat visualisasi dan simpan ke folder stats/<video_name>/
    visualize_stats(stats, {
        'videos': config['output_dirs']['videos'],
        'stats': video_stats_dir
    }, config['models'])

    print(f"\nSelesai! Hasil inferensi dan statistik disimpan di:\n{video_stats_dir}")



if __name__ == "__main__":
    main()