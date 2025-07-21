import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_yaml(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_class_distribution(dataset_path, classes):
    """Get distribution of classes in the dataset."""
    class_counts = {cls: 0 for cls in classes}

    labels_path = os.path.join(dataset_path, 'labels')
    if not os.path.exists(labels_path):
        return class_counts

    for label_file in os.listdir(labels_path):
        if not label_file.endswith('.txt'):
            continue

        file_path = os.path.join(labels_path, label_file)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.strip():
                class_id = int(line.strip().split()[0])
                if class_id < len(classes):
                    class_counts[classes[class_id]] += 1

    return class_counts


def plot_combined_distribution(distributions, splits, classes, output_dir):
    """Plot combined class distribution for all splits in a single figure."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a single combined plot with multiple subplots
    fig, axes = plt.subplots(len(splits) + 1, 1, figsize=(12, 4 * (len(splits) + 1)))

    # Create bar plots for each split
    for i, split in enumerate(splits):
        counts = [distributions[split][cls] for cls in classes]
        bars = axes[i].bar(classes, counts, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(classes)])
        axes[i].set_title(f'{split.capitalize()} Split Distribution')
        axes[i].set_ylabel('Count')

        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom')

    # Combined plot for all splits
    x = np.arange(len(classes))
    bar_width = 0.8 / len(splits)

    for i, split in enumerate(splits):
        counts = [distributions[split][cls] for cls in classes]
        bars = axes[-1].bar(x + i * bar_width - 0.4 + bar_width / 2, counts, width=bar_width,
                            label=split.capitalize(), alpha=0.7)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if there's a value
                axes[-1].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                              f'{int(height)}', ha='center', va='bottom', fontsize=8)

    axes[-1].set_title('Combined Class Distribution Across Splits')
    axes[-1].set_ylabel('Count')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(classes)
    axes[-1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()


def visualize_samples(dataset_base, splits, classes, num_samples, output_dir):
    """Visualize random samples from each split with bounding boxes and counts."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a single figure for all split samples
    rows = len(splits)
    cols = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Handle the case when there's only one sample or one split
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for row, split in enumerate(splits):
        split_path = os.path.join(dataset_base, split)
        images_path = os.path.join(split_path, 'images')
        labels_path = os.path.join(split_path, 'labels')

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            for col in range(cols):
                axes[row, col].text(0.5, 0.5, f"No images in {split}",
                                    horizontalalignment='center', verticalalignment='center')
                axes[row, col].axis('off')
            continue

        # Get all image files
        image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            for col in range(cols):
                axes[row, col].text(0.5, 0.5, f"No images in {split}",
                                    horizontalalignment='center', verticalalignment='center')
                axes[row, col].axis('off')
            continue

        # Sample random images
        samples = random.sample(image_files, min(num_samples, len(image_files)))

        for col, sample in enumerate(samples):
            img_path = os.path.join(images_path, sample)
            label_path = os.path.join(labels_path, os.path.splitext(sample)[0] + '.txt')

            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, _ = img.shape

            # Initialize class counts for this image
            image_class_counts = {cls: 0 for cls in classes}

            # Load and draw bounding boxes
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        class_id = int(parts[0])

                        if class_id < len(classes):
                            class_name = classes[class_id]
                            image_class_counts[class_name] += 1

                            x_center, y_center, box_width, box_height = map(float, parts[1:5])

                            # Convert to pixel coordinates
                            x1 = int((x_center - box_width / 2) * w)
                            y1 = int((y_center - box_height / 2) * h)
                            x2 = int((x_center + box_width / 2) * w)
                            y2 = int((y_center + box_height / 2) * h)

                            # Draw rectangle (different color for each class)
                            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                            color = colors[class_id % len(colors)]
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                            # Add label
                            cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            axes[row, col].imshow(img)

            # Create title with counts
            count_str = ", ".join([f"{cls}: {count}" for cls, count in image_class_counts.items() if count > 0])
            title = f"{split.capitalize()} sample {col + 1}\n{count_str}"
            axes[row, col].set_title(title)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_visualization.png'))
    plt.close()


def save_statistics_to_yaml(distributions, splits, classes, output_dir):
    """Save statistics to a YAML file instead of printing to terminal."""
    statistics = {}

    # Per split statistics
    split_stats = {}
    for split in splits:
        total_split = sum(distributions[split].values())
        class_stats = {}

        for cls in classes:
            count = distributions[split][cls]
            percentage = (count / total_split * 100) if total_split > 0 else 0
            class_stats[cls] = {
                'count': count,
                'percentage': f"{percentage:.2f}%"
            }

        split_stats[split] = {
            'total_instances': total_split,
            'classes': class_stats
        }

    # Combined statistics
    total_instances = {cls: sum(distributions[split][cls] for split in splits) for cls in classes}
    total_all = sum(total_instances.values())

    combined_stats = {
        'total_instances': total_all,
        'classes': {}
    }

    for cls in classes:
        count = total_instances[cls]
        percentage = (count / total_all * 100) if total_all > 0 else 0
        combined_stats['classes'][cls] = {
            'count': count,
            'percentage': f"{percentage:.2f}%"
        }

    # Combine all statistics
    statistics = {
        'splits': split_stats,
        'combined': combined_stats
    }

    # Save to YAML file
    with open(os.path.join(output_dir, 'distribution.yaml'), 'w') as f:
        yaml.dump(statistics, f, sort_keys=False, default_flow_style=False)


def analyze_dataset(config):
    """Analyze dataset based on configuration."""
    dataset_base = config['dataset_base']
    data_yaml_path = config['data_yaml_path']
    splits = config['splits']
    num_samples = config['num_samples_per_split']
    output_dir = config['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    # Load YAML file to get class information
    yaml_data = load_yaml(data_yaml_path)
    classes = yaml_data.get('names', [])

    # Get class distribution for each split
    distributions = {}

    for split in splits:
        split_path = os.path.join(dataset_base, split)

        if not os.path.exists(split_path):
            distributions[split] = {cls: 0 for cls in classes}
            continue

        distributions[split] = get_class_distribution(split_path, classes)

    # Plot distributions
    plot_combined_distribution(distributions, splits, classes, output_dir)

    # Visualize samples
    visualize_samples(dataset_base, splits, classes, num_samples, output_dir)

    # Save statistics to YAML file
    save_statistics_to_yaml(distributions, splits, classes, output_dir)


if __name__ == "__main__":
    CONFIG = {
        'dataset_base': '../dataset/HOME-FIRE',
        'data_yaml_path': '../dataset/HOME-FIRE/data.yaml',
        'splits': ['train', 'val', 'test'],
        'target_classes': ['Fire', 'Smoke'],
        'num_samples_per_split': 2,
        'output_dir': '../utils/figures/distribution'
    }

    analyze_dataset(CONFIG)