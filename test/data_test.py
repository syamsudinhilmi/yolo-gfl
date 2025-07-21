import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou


def load_model_config(config_path):
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_test_images(image_config):
    """Get all test images from configuration"""
    image_dir = image_config['path']
    label_dir = image_dir.replace('images', 'labels')

    # Get all image files
    image_files = [f for f in os.listdir(image_dir)
                   if f.endswith(('.jpg', '.jpeg', '.png')) and
                   os.path.exists(os.path.join(label_dir, f.rsplit('.', 1)[0] + '.txt'))]

    # Create list of image paths and corresponding label paths
    selected_data = []
    for img_file in tqdm(image_files, desc="Listing images"):
        base_name = img_file.rsplit('.', 1)[0]
        label_file = base_name + '.txt'

        selected_data.append({
            'image_path': os.path.join(image_dir, img_file),
            'label_path': os.path.join(label_dir, label_file),
            'name': base_name
        })

    print(f"Found {len(selected_data)} images in test set")
    return selected_data


def load_ground_truth(label_path, img_width, img_height):
    """Load ground truth labels from YOLO format txt file"""
    gt_boxes = []
    gt_classes = []

    try:
        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:  # class, x_center, y_center, width, height
                    cls = int(data[0])
                    # Convert from normalized coordinates to absolute
                    x_center = float(data[1]) * img_width
                    y_center = float(data[2]) * img_height
                    width = float(data[3]) * img_width
                    height = float(data[4]) * img_height

                    # Convert from center coordinates to corner coordinates
                    x1 = x_center - (width / 2)
                    y1 = y_center - (height / 2)
                    x2 = x_center + (width / 2)
                    y2 = y_center + (height / 2)

                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(cls)
    except FileNotFoundError:
        print(f"Warning: Label file not found {label_path}")

    return np.array(gt_boxes), np.array(gt_classes)


def process_image(model, image_info, model_config, output_dir, stats):
    """Process a single image with the model and compare with ground truth"""
    # Read image
    try:
        image = cv2.imread(image_info['image_path'])
        if image is None:
            print(f"Error: Could not read image {image_info['name']}")
            return None

        img_height, img_width = image.shape[:2]
    except Exception as e:
        print(f"Error loading image {image_info['name']}: {str(e)}")
        return None

    # Load ground truth
    try:
        gt_boxes, gt_classes = load_ground_truth(image_info['label_path'], img_width, img_height)
    except Exception as e:
        print(f"Error loading labels for {image_info['name']}: {str(e)}")
        gt_boxes, gt_classes = np.array([]), np.array([])

    # Run inference
    start_time = time.time()
    try:
        results = model(image, conf=model_config['conf_threshold'], verbose=False)
        inference_time = time.time() - start_time
    except Exception as e:
        print(f"Error during inference on {image_info['name']}: {str(e)}")
        return None

    # Extract predictions
    pred_boxes = []
    pred_classes = []
    pred_scores = []

    if results and results[0].boxes:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()  # Get bounding box in [x1, y1, x2, y2] format

            pred_boxes.append(xyxy)
            pred_classes.append(cls)
            pred_scores.append(conf)

    # Convert to numpy arrays
    pred_boxes = np.array(pred_boxes) if pred_boxes else np.array([])
    pred_classes = np.array(pred_classes) if pred_classes else np.array([])
    pred_scores = np.array(pred_scores) if pred_scores else np.array([])

    # Calculate metrics
    metrics = calculate_metrics(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores)

    # Save annotated image
    model_img_dir = os.path.join(output_dir['data_test'], model_config['name'])
    os.makedirs(model_img_dir, exist_ok=True)

    try:
        annotated_img = results[0].plot()
        output_path = os.path.join(model_img_dir, f"{image_info['name']}.jpg")
        cv2.imwrite(output_path, annotated_img)
    except Exception as e:
        print(f"Error saving annotated image {image_info['name']}: {str(e)}")

    # Store stats
    stats[model_config['name']][image_info['name']] = {
        'inference_time': inference_time,
        'num_predictions': len(pred_boxes),
        'num_gt': len(gt_boxes),
        'metrics': metrics,
        'fire_count': sum(1 for cls in pred_classes if cls == 1),
        'smoke_count': sum(1 for cls in pred_classes if cls == 0)
    }

    return inference_time


def calculate_fps(model, model_config, test_images, warmup=10, runs=100):
    """Calculate FPS (Frames Per Second) for the model"""
    print(f"\nCalculating FPS for {model_config['name']}...")

    # Warmup runs
    for _ in range(warmup):
        for image_info in test_images[:5]:  # Use first 5 images for warmup
            try:
                image = cv2.imread(image_info['image_path'])
                if image is not None:
                    _ = model(image, conf=model_config['conf_threshold'], verbose=False)
            except:
                pass

    # Actual FPS measurement
    total_time = 0
    for _ in tqdm(range(runs), desc="FPS Test"):
        for image_info in test_images[:5]:  # Use first 5 images for consistent measurement
            try:
                image = cv2.imread(image_info['image_path'])
                if image is not None:
                    start_time = time.time()
                    _ = model(image, conf=model_config['conf_threshold'], verbose=False)
                    total_time += time.time() - start_time
            except:
                pass

    if total_time > 0:
        fps = (runs * 5) / total_time  # 5 images per run
    else:
        fps = 0

    print(f"  FPS: {fps:.2f}")
    return fps


def calculate_metrics(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_threshold=0.5):
    """Calculate precision, recall and F1 score"""
    # Initialize metrics
    metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'class_accuracy': 0.0,
        'matches': []
    }

    # If no ground truth or predictions, return default metrics
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            metrics['precision'] = 1.0
            metrics['recall'] = 1.0
            metrics['f1_score'] = 1.0
        return metrics

    # Calculate IoU between each prediction and ground truth
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        gt_boxes_tensor = torch.tensor(gt_boxes)
        pred_boxes_tensor = torch.tensor(pred_boxes)

        iou_matrix = box_iou(pred_boxes_tensor, gt_boxes_tensor)
        iou_values = iou_matrix.numpy()

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        matches = []

        # Sort predictions by confidence
        pred_indices = np.argsort(-pred_scores)

        for pred_idx in pred_indices:
            # Find best matching ground truth for this prediction
            best_iou = iou_threshold
            best_gt_idx = -1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue

                iou = iou_values[pred_idx, gt_idx]

                if iou >= best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                # Match found
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

                class_match = pred_classes[pred_idx] == gt_classes[best_gt_idx]

                matches.append({
                    'pred_idx': int(pred_idx),
                    'gt_idx': int(best_gt_idx),
                    'iou': float(best_iou),
                    'pred_class': int(pred_classes[pred_idx]),
                    'gt_class': int(gt_classes[best_gt_idx]),
                    'class_match': bool(class_match)
                })

        # Calculate metrics
        tp = len(matched_pred)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)

        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
                metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0

        # Calculate class accuracy for matched boxes
        class_correct = sum(1 for m in matches if m['class_match'])
        metrics['class_accuracy'] = class_correct / len(matches) if len(matches) > 0 else 0.0
        metrics['matches'] = matches

    return metrics


def visualize_results(stats, output_dir, model_configs):
    """Generate visualizations of results"""
    # Create the data_test directory inside stats
    img_inference_dir = os.path.join(output_dir['stats'], 'data_test')
    os.makedirs(img_inference_dir, exist_ok=True)

    # 1. Precision, Recall, F1 by model
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)

    model_names = list(stats.keys())
    metrics_avg = {name: {'precision': [], 'recall': [], 'f1_score': []} for name in model_names}

    # Collect metrics across all images
    for model_name in model_names:
        for img_name, img_stats in stats[model_name].items():
            if 'metrics' in img_stats:
                metrics_avg[model_name]['precision'].append(img_stats['metrics']['precision'])
                metrics_avg[model_name]['recall'].append(img_stats['metrics']['recall'])
                metrics_avg[model_name]['f1_score'].append(img_stats['metrics']['f1_score'])

    # Average metrics
    x = np.arange(len(model_names))
    width = 0.25

    precisions = [np.mean(metrics_avg[name]['precision']) for name in model_names]
    recalls = [np.mean(metrics_avg[name]['recall']) for name in model_names]
    f1_scores = [np.mean(metrics_avg[name]['f1_score']) for name in model_names]

    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1 Score')

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Detection Performance Metrics')
    plt.xticks(x, model_names)
    plt.legend()

    # 2. Inference time comparison (in milliseconds)
    plt.subplot(2, 2, 2)

    avg_times = []
    std_times = []
    avg_times_ms = []  # Store millisecond times for display
    std_times_ms = []  # Store millisecond standard deviations

    for model_name in model_names:
        times = [img_stats['inference_time'] for img_stats in stats[model_name].values()
                 if 'inference_time' in img_stats]
        if times:
            avg_times.append(np.mean(times))
            std_times.append(np.std(times))
        else:
            avg_times.append(0)
            std_times.append(0)

        # Convert to milliseconds
        avg_times_ms.append(np.mean(times) * 1000 if times else 0)
        std_times_ms.append(np.std(times) * 1000 if times else 0)

    plt.bar(x, avg_times_ms, yerr=std_times_ms, capsize=5)
    plt.xlabel('Model')
    plt.ylabel('Time (milliseconds)')
    plt.title('Average Inference Time')
    plt.xticks(x, model_names)

    # Add exact millisecond values on top of bars
    for i, v in enumerate(avg_times_ms):
        plt.text(i, v + std_times_ms[i], f"{v:.2f} ms", ha='center', va='bottom')

    # 3. Detection counts by class
    plt.subplot(2, 2, 3)

    fire_counts = []
    smoke_counts = []

    for model_name in model_names:
        fire_counts.append(sum(img_stats.get('fire_count', 0) for img_stats in stats[model_name].values()))
        smoke_counts.append(sum(img_stats.get('smoke_count', 0) for img_stats in stats[model_name].values()))

    width = 0.35
    plt.bar(x - width / 2, fire_counts, width, label='Fire')
    plt.bar(x + width / 2, smoke_counts, width, label='Smoke')

    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.title('Detection Counts by Class')
    plt.xticks(x, model_names)
    plt.legend()

    # 4. Model Size vs Performance
    plt.subplot(2, 2, 4)

    model_sizes = [config.get('size_mb', 0) for config in model_configs]
    model_f1 = [np.mean(metrics_avg[name]['f1_score']) for name in model_names]
    model_fps = [config.get('fps', 0) for config in model_configs]  # Add FPS to the plot

    # Create a scatter plot with size representing FPS
    scatter = plt.scatter(model_sizes, model_f1, s=[fps * 5 for fps in model_fps], alpha=0.5)

    # Add labels with model names and FPS values
    for i, name in enumerate(model_names):
        plt.annotate(f"{name}\n{model_fps[i]:.1f} FPS",
                     (model_sizes[i], model_f1[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.xlabel('Model Size (MB)')
    plt.ylabel('F1 Score')
    plt.title('Model Size vs. Performance (Bubble size = FPS)')

    plt.tight_layout()
    plt.savefig(os.path.join(img_inference_dir, 'image_test_performance.png'), dpi=300)

    # 5. Per-class metrics
    plt.figure(figsize=(12, 6))

    # Confusion matrix-like visualization
    confusion_data = {}
    for model_name in model_names:
        confusion_data[model_name] = {
            'smoke_as_smoke': 0,
            'smoke_as_fire': 0,
            'fire_as_fire': 0,
            'fire_as_smoke': 0
        }

        for img_name, img_stats in stats[model_name].items():
            if 'metrics' in img_stats and 'matches' in img_stats['metrics']:
                for match in img_stats['metrics']['matches']:
                    if match['gt_class'] == 0:  # Smoke
                        if match['pred_class'] == 0:
                            confusion_data[model_name]['smoke_as_smoke'] += 1
                        else:
                            confusion_data[model_name]['smoke_as_fire'] += 1
                    else:  # Fire
                        if match['pred_class'] == 1:
                            confusion_data[model_name]['fire_as_fire'] += 1
                        else:
                            confusion_data[model_name]['fire_as_smoke'] += 1

    # Plot per-model confusion matrix
    for i, model_name in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)

        cm = np.array([
            [confusion_data[model_name]['smoke_as_smoke'], confusion_data[model_name]['smoke_as_fire']],
            [confusion_data[model_name]['fire_as_smoke'], confusion_data[model_name]['fire_as_fire']]
        ])

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_name} Class Predictions')
        plt.colorbar()

        classes = ['Smoke', 'Fire']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')

    plt.tight_layout()
    plt.savefig(os.path.join(img_inference_dir, 'class_confusion.png'), dpi=300)

    # NEW: Add average inference time table
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(111)
    ax.axis('off')

    # Create data for table - now includes FPS
    table_data = []
    for i, model_name in enumerate(model_names):
        avg_time_ms = avg_times_ms[i]
        std_time_ms = std_times_ms[i]
        fps = model_configs[i].get('fps', 0)
        table_data.append([model_name, f"{avg_time_ms:.2f} ± {std_time_ms:.2f}", f"{fps:.2f}"])

    # Add table
    table = plt.table(
        cellText=table_data,
        colLabels=['Model', 'Inference Time (ms)', 'FPS'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(img_inference_dir, 'performance_table.png'), dpi=300)

    # NEW: Add additional per-class performance metrics
    plt.figure(figsize=(12, 8))

    # Calculate class-specific metrics
    class_metrics = {}
    for model_name in model_names:
        class_metrics[model_name] = {
            'smoke_precision': [],
            'smoke_recall': [],
            'fire_precision': [],
            'fire_recall': []
        }

        for img_name, img_stats in stats[model_name].items():
            if 'metrics' not in img_stats:
                continue

            gt_smoke = 0
            gt_fire = 0
            tp_smoke = 0
            tp_fire = 0
            fp_smoke = 0
            fp_fire = 0

            if 'matches' in img_stats['metrics']:
                for match in img_stats['metrics']['matches']:
                    if match['gt_class'] == 0:  # Ground truth is smoke
                        gt_smoke += 1
                        if match['pred_class'] == 0:  # Predicted as smoke
                            tp_smoke += 1
                    elif match['gt_class'] == 1:  # Ground truth is fire
                        gt_fire += 1
                        if match['pred_class'] == 1:  # Predicted as fire
                            tp_fire += 1

            # Count false positives
            if 'pred_classes' in img_stats['metrics']:
                for cls in img_stats['metrics']['pred_classes']:
                    if cls == 0 and not any(
                            m['pred_class'] == 0 and m['class_match'] for m in img_stats['metrics'].get('matches', [])):
                        fp_smoke += 1
                    elif cls == 1 and not any(
                            m['pred_class'] == 1 and m['class_match'] for m in img_stats['metrics'].get('matches', [])):
                        fp_fire += 1

            # Calculate metrics for this image
            smoke_precision = tp_smoke / (tp_smoke + fp_smoke) if (tp_smoke + fp_smoke) > 0 else 0
            smoke_recall = tp_smoke / gt_smoke if gt_smoke > 0 else 0
            fire_precision = tp_fire / (tp_fire + fp_fire) if (tp_fire + fp_fire) > 0 else 0
            fire_recall = tp_fire / gt_fire if gt_fire > 0 else 0

            class_metrics[model_name]['smoke_precision'].append(smoke_precision)
            class_metrics[model_name]['smoke_recall'].append(smoke_recall)
            class_metrics[model_name]['fire_precision'].append(fire_precision)
            class_metrics[model_name]['fire_recall'].append(fire_recall)

    # Plot class-specific metrics
    width = 0.1
    plt.subplot(2, 1, 1)

    for i, model_name in enumerate(model_names):
        smoke_precision = np.nanmean(class_metrics[model_name]['smoke_precision'])
        smoke_recall = np.nanmean(class_metrics[model_name]['smoke_recall'])
        fire_precision = np.nanmean(class_metrics[model_name]['fire_precision'])
        fire_recall = np.nanmean(class_metrics[model_name]['fire_recall'])

        # Position for each group of bars
        base_pos = i * (width * 4 + 0.2)

        # Plot bars
        plt.bar(base_pos, smoke_precision, width, label=f'{model_name} Smoke Precision' if i == 0 else "")
        plt.bar(base_pos + width, smoke_recall, width, label=f'{model_name} Smoke Recall' if i == 0 else "")
        plt.bar(base_pos + 2 * width, fire_precision, width, label=f'{model_name} Fire Precision' if i == 0 else "")
        plt.bar(base_pos + 3 * width, fire_recall, width, label=f'{model_name} Fire Recall' if i == 0 else "")

        # Add text above each bar
        plt.text(base_pos, smoke_precision + 0.02, f"{smoke_precision:.2f}", ha='center', va='bottom', fontsize=8)
        plt.text(base_pos + width, smoke_recall + 0.02, f"{smoke_recall:.2f}", ha='center', va='bottom', fontsize=8)
        plt.text(base_pos + 2 * width, fire_precision + 0.02, f"{fire_precision:.2f}", ha='center', va='bottom',
                 fontsize=8)
        plt.text(base_pos + 3 * width, fire_recall + 0.02, f"{fire_recall:.2f}", ha='center', va='bottom', fontsize=8)

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Per-Class Detection Performance')
    plt.xticks([i * (width * 4 + 0.2) + 1.5 * width for i in range(len(model_names))], model_names)
    plt.legend(ncol=2)
    plt.ylim(0, 1.1)

    # Save detailed stats to YAML
    detailed_stats = {}
    for model_name in model_names:
        inference_times = [img_stats.get('inference_time', 0) for img_stats in stats[model_name].values()]
        detailed_stats[model_name] = {
            'avg_inference_time_ms': float(np.nanmean(inference_times) * 1000),
            'std_inference_time_ms': float(np.nanstd(inference_times) * 1000),
            'precision': float(np.nanmean(metrics_avg[model_name]['precision'])),
            'recall': float(np.nanmean(metrics_avg[model_name]['recall'])),
            'f1_score': float(np.nanmean(metrics_avg[model_name]['f1_score'])),
            'total_fire_detections': int(
                sum(img_stats.get('fire_count', 0) for img_stats in stats[model_name].values())),
            'total_smoke_detections': int(
                sum(img_stats.get('smoke_count', 0) for img_stats in stats[model_name].values())),
            'smoke_precision': float(np.nanmean(class_metrics[model_name]['smoke_precision'])),
            'smoke_recall': float(np.nanmean(class_metrics[model_name]['smoke_recall'])),
            'fire_precision': float(np.nanmean(class_metrics[model_name]['fire_precision'])),
            'fire_recall': float(np.nanmean(class_metrics[model_name]['fire_recall'])),
            'fps': float(model_configs[i].get('fps', 0))  # Add FPS to stats
        }

    # Plot F1 score
    plt.subplot(2, 1, 2)

    smoke_f1 = []
    fire_f1 = []

    for model_name in model_names:
        smoke_p = np.nanmean(class_metrics[model_name]['smoke_precision'])
        smoke_r = np.nanmean(class_metrics[model_name]['smoke_recall'])
        fire_p = np.nanmean(class_metrics[model_name]['fire_precision'])
        fire_r = np.nanmean(class_metrics[model_name]['fire_recall'])

        # Calculate F1 scores
        smoke_f1_score = 2 * (smoke_p * smoke_r) / (smoke_p + smoke_r) if (smoke_p + smoke_r) > 0 else 0
        fire_f1_score = 2 * (fire_p * fire_r) / (fire_p + fire_r) if (fire_p + fire_r) > 0 else 0

        smoke_f1.append(smoke_f1_score)
        fire_f1.append(fire_f1_score)

        # Add to detailed stats
        detailed_stats[model_name]['smoke_f1_score'] = float(smoke_f1_score)
        detailed_stats[model_name]['fire_f1_score'] = float(fire_f1_score)

    # Plot F1 scores
    width = 0.35
    x = np.arange(len(model_names))
    plt.bar(x - width / 2, smoke_f1, width, label='Smoke F1')
    plt.bar(x + width / 2, fire_f1, width, label='Fire F1')

    # Add text above each bar
    for i, (s_f1, f_f1) in enumerate(zip(smoke_f1, fire_f1)):
        plt.text(i - width / 2, s_f1 + 0.02, f"{s_f1:.2f}", ha='center', va='bottom')
        plt.text(i + width / 2, f_f1 + 0.02, f"{f_f1:.2f}", ha='center', va='bottom')

    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(img_inference_dir, 'class_specific_metrics.png'), dpi=300)

    # Save filtered stats to YAML (without large data)
    filtered_stats = {}
    for model_name, imgs in stats.items():
        filtered_stats[model_name] = {}

        # Calculate average inference time in milliseconds
        inference_times = [img_stat.get('inference_time', 0) for img_stat in imgs.values()]
        avg_inference_time_ms = np.mean(inference_times) * 1000 if inference_times else 0

        # Add average inference time to model stats
        filtered_stats[model_name]['avg_inference_time_ms'] = float(avg_inference_time_ms)

        for img_name, img_stat in imgs.items():
            # Convert inference time to milliseconds
            img_stat_copy = img_stat.copy()
            if 'inference_time' in img_stat_copy:
                img_stat_copy['inference_time_ms'] = img_stat_copy['inference_time'] * 1000
                del img_stat_copy['inference_time']

            # Remove the detailed match info to keep file size reasonable
            if 'metrics' in img_stat_copy and 'matches' in img_stat_copy['metrics']:
                img_stat_copy['metrics'] = {k: v for k, v in img_stat_copy['metrics'].items() if k != 'matches'}

            filtered_stats[model_name][img_name] = img_stat_copy

    # Save stats
    with open(os.path.join(img_inference_dir, 'image_test_stats.yaml'), 'w') as f:
        yaml.dump(filtered_stats, f)

    # Save the detailed model comparison stats
    with open(os.path.join(img_inference_dir, 'model_comparison.yaml'), 'w') as f:
        yaml.dump(detailed_stats, f)
    return img_inference_dir


def main():
    # Configuration path
    CONFIG_PATH = "../config/test_cfg.yaml"

    # Load configuration
    config = load_model_config(CONFIG_PATH)

    # Create output directories
    for dir_key, dir_path in config['output_dirs'].items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Check if images section exists in config
    if 'images' not in config or not config['images']:
        print("No image configurations found in config file. Exiting.")
        return

    # Select test images - get ALL images
    image_config = config['images'][0]  # Use the first image configuration
    test_images = get_test_images(image_config)

    # Initialize stats collection
    stats = {model['name']: {} for model in config['models']}

    # Process each image with each model
    for model_config in config['models']:
        print(f"\nProcessing Model: {model_config['name']}")

        # Calculate model size
        try:
            model_size_bytes = os.path.getsize(model_config['path'])
            model_config['size_mb'] = model_size_bytes / (1024 * 1024)
        except Exception as e:
            print(f"Error getting model size: {str(e)}")
            model_config['size_mb'] = 0

        # Load model
        try:
            model = YOLO(model_config['path'])
            print(f"Loaded model: {model_config['name']} ({model_config['size_mb']:.2f} MB)")
        except Exception as e:
            print(f"Error loading model {model_config['name']}: {str(e)}")
            continue

        # Calculate FPS before processing images
        fps = calculate_fps(model, model_config, test_images)
        model_config['fps'] = fps

        # Process images with progress bar
        total_inference_time = 0
        start_total_time = time.time()

        for image_info in tqdm(test_images, desc=f"Processing {model_config['name']}"):
            inf_time = process_image(
                model=model,
                image_info=image_info,
                model_config=model_config,
                output_dir=config['output_dirs'],
                stats=stats
            )
            if inf_time:
                total_inference_time += inf_time

        end_total_time = time.time()
        elapsed_total_time = end_total_time - start_total_time

        print(f"  Completed {len(test_images)} images in {elapsed_total_time:.2f} seconds "
              f"(total inference time: {total_inference_time:.2f} seconds)")

    # Generate visualizations
    visualize_results(stats, config['output_dirs'], config['models'])
    print("\nProcessing complete! Check output directories for results.")


if __name__ == "__main__":
    main()