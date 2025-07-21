import time

from ultralytics import YOLO


def evaluate_model(model_path: str, data_path: str, project_name: str, run_name: str):
    """Original evaluation function"""
    print(f"\nEvaluating Model: {run_name}")
    model = YOLO(model_path, task='detect')
    print(model.info())

    metrics = model.val(
        data=data_path,
        save_json=True,
        project=project_name,
        name=run_name,
    )
    return metrics

def main():
    data_path = r"../dataset/HOME-FIRE/data.yaml"

    models_to_evaluate = [
        {
            "model_path": r'../yolov12/runs/YOLOv12/weights/best.pt',
            "project_name": 'runs',
            "run_name": 'YOLO12',
        },
        {
            "model_path": r'../yolo-gfl/runs/YOLO-GFL/weights/best.pt',
            "project_name": 'runs',
            "run_name": 'YOLO-GFL',
        },
    ]

    # Test different image sizes (optional)
    img_sizes_to_test = [640]  # Add more sizes like [320, 640, 1280] if needed

    all_results = []

    print(f"\nSTARTING COMPREHENSIVE MODEL TESTING")

    for model_info in models_to_evaluate:
        model_path = model_info["model_path"]
        project_name = model_info["project_name"]
        run_name = model_info["run_name"]

        for img_size in img_sizes_to_test:
            print(f"\nTesting {run_name} with image size {img_size}")
            start_time = time.time()
            metrics = evaluate_model(model_path, data_path, project_name, run_name)
            elapsed_time = time.time() - start_time

            # Store results
            result = {
                "model": run_name,
                "img_size": img_size,
                "metrics": metrics,
                "elapsed_time": elapsed_time
            }
            all_results.append(result)

if __name__ == "__main__":
    main()