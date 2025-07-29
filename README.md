# <center>YOLO-GFL</center>

**YOLO Architecture Optimization Using GhostNet-Based Modules for Indoor Fire and Smoke Detection**

## Overview

YOLO-GFL is an optimized YOLO architecture that incorporates GhostNet-based modules for efficient indoor fire and smoke detection. This implementation provides a lightweight yet accurate solution for real-time fire safety monitoring systems.

## Performance Comparison

### Benchmarks: YOLO-v12n vs YOLO-GFL

![Performance Analysis](test/runs_test/performance_analysis/time_vs_fps_comparison.png)

## Model Specifications

### Dataset Information
- **Source**: [PENG BO Home Fire Dataset](https://github.com/PengBo0/Home-fire-dataset)
- **Baseline Model**: [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- **Test Set**: 1,300 images with 1,601 total instances
  - Fire instances: 897
  - Smoke instances: 689
  - Background instances: 15

### Model Comparison

| Model | Layers | Parameters | GFLOPs | Model Size |
|-------|--------|------------|---------|------------|
| **YOLO-GFL** | 118 | 1,611,126 | 4.6 | **3.36 MB** |
| **YOLOv12** | 159 | 2,527,166 | 5.8 | 5.21 MB |

## Results

### Test Set Performance

#### YOLO-GFL Results
```
Class     Images  Instances  Precision  Recall  mAP50   mAP50-95
All       1300    1586       0.908      0.816   0.891   0.564
Fire      852     897        0.929      0.818   0.914   0.590
Smoke     618     689        0.887      0.813   0.868   0.539
```

#### YOLOv12 Results
```
Class     Images  Instances  Precision  Recall  mAP50   mAP50-95
All       1300    1586       0.901      0.834   0.894   0.570
Fire      852     897        0.923      0.851   0.917   0.592
Smoke     618     689        0.878      0.816   0.872   0.548
```

### Training & Validation Performance (300 epochs)

#### YOLO-GFL Training Results
```
Class     Images  Instances  Precision  Recall  mAP50   mAP50-95
All       1300    1580       0.940      0.892   0.932   0.621
Fire      894     963        0.955      0.918   0.954   0.671
Smoke     574     617        0.925      0.865   0.910   0.572
```

#### YOLOv12 Training Results
```
Class     Images  Instances  Precision  Recall  mAP50   mAP50-95
All       1300    1580       0.937      0.886   0.932   0.642
Fire      894     963        0.959      0.920   0.955   0.695
Smoke     574     617        0.916      0.853   0.909   0.588
```

## Demo Results

### Fire Detection Comparison

| Test Image   | YOLO-GFL                                                                                     | YOLOv12                                                                                    |
|--------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Test 1       | ![YOLO-GFL Result 1](test/output/images/batch_20250729_104133/output/imgtest_1_YOLO-GFL.png) | ![YOLOv12 Result 1](test/output/images/batch_20250729_104133/output/imgtest_1_YOLOv12.png) |
| Test 2       | ![YOLO-GFL Result 2](test/output/images/batch_20250729_104133/output/imgtest_2_YOLO-GFL.png) | ![YOLOv12 Result 2](test/output/images/batch_20250729_104133/output/imgtest_2_YOLOv12.png) |
| Test 3       | ![YOLO-GFL Result 3](test/output/images/batch_20250729_104133/output/imgtest_3_YOLO-GFL.png) | ![YOLOv12 Result 3](test/output/images/batch_20250729_104133/output/imgtest_3_YOLOv12.png) |
| Test 4       | ![YOLO-GFL Result 4](test/output/images/batch_20250729_104133/output/imgtest_4_YOLO-GFL.png) | ![YOLOv12 Result 4](test/output/images/batch_20250729_104133/output/imgtest_4_YOLOv12.png) |
| Test 5       | ![YOLO-GFL Result 5](test/output/images/batch_20250729_104133/output/imgtest_5_YOLO-GFL.png) | ![YOLOv12 Result 5](test/output/images/batch_20250729_104133/output/imgtest_5_YOLOv12.png) |
| Test 6       | ![YOLO-GFL Result 6](test/output/images/batch_20250729_104133/output/imgtest_6_YOLO-GFL.png) | ![YOLOv12 Result 6](test/output/images/batch_20250729_104133/output/imgtest_6_YOLOv12.png) |
| Test 7       | ![YOLO-GFL Result 7](test/output/images/batch_20250729_104133/output/imgtest_7_YOLO-GFL.png) | ![YOLOv12 Result 7](test/output/images/batch_20250729_104133/output/imgtest_7_YOLOv12.png) |
| Test 8       | ![YOLO-GFL Result 8](test/output/images/batch_20250729_104133/output/imgtest_8_YOLO-GFL.png) | ![YOLOv12 Result 8](test/output/images/batch_20250729_104133/output/imgtest_8_YOLOv12.png) |

## Analysis & Visualizations

### Confusion Matrices

#### Test Set Evaluation
- **YOLO-GFL**: ![Confusion Matrix](test/runs_test/evaluation/YOLO-GFL/confusion_matrix.png)
- **YOLOv12**: ![Confusion Matrix](test/runs_test/evaluation/YOLOv12/confusion_matrix.png)

#### Training Validation Set
- **YOLO-GFL**: ![Training Confusion Matrix](yolo-gfl/runs/YOLO-GFL/confusion_matrix.png)
- **YOLOv12**: ![Training Confusion Matrix](yolov12/runs/YOLOv12/confusion_matrix.png)

### Training Progress

#### YOLO-GFL Training Metrics
![YOLO-GFL Results](yolo-gfl/runs/YOLO-GFL/results.png)

#### YOLOv12 Training Metrics
![YOLOv12 Results](yolov12/runs/YOLOv12/results.png)

## Key Advantages of YOLO-GFL

1. **Lightweight Architecture**: 35% smaller model size (3.36 MB vs 5.21 MB)
2. **Efficient Computation**: 21% reduction in GFLOPs (4.6 vs 5.8)
3. **Competitive Accuracy**: Maintains comparable detection performance
4. **Real-time Capability**: Optimized for deployment in resource-constrained environments
5. **Fire Safety Focus**: Specialized for indoor fire and smoke detection scenarios

## Technical Implementation

The YOLO-GFL architecture integrates GhostNet-based modules to achieve model compression while maintaining detection accuracy. The optimization focuses on reducing computational complexity and memory footprint, making it suitable for edge deployment in fire safety systems.

[//]: # (## Citation)

[//]: # ()
[//]: # (If you use this work in your research, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{yolo-gfl,)

[//]: # (  title={YOLO Architecture Optimization Using GhostNet-Based Modules for Indoor Fire and Smoke Detection},)

[//]: # (  author={[Your Name]},)

[//]: # (  journal={[Journal Name]},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

## Acknowledgments

- Dataset: [PENG BO Home Fire Dataset](https://github.com/PengBo0/Home-fire-dataset)
- Baseline Model: [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- GhostNet Architecture for efficient module design