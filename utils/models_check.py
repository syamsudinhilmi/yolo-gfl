import torch
import os
from thop import profile, clever_format
from ultralytics import YOLO
import pandas as pd


def create_descriptive_name(model_path):
    # Normalize path
    normalized_path = os.path.normpath(model_path)
    path_parts = normalized_path.split(os.sep)

    # Cari folder runs untuk mengambil nama eksperimen
    if 'runs' in path_parts:
        runs_index = path_parts.index('runs')
        if runs_index + 1 < len(path_parts):
            experiment_name = path_parts[runs_index + 1]
            return f"{experiment_name}_best.pt"

    # Jika tidak ada folder runs, gunakan folder parent
    if len(path_parts) >= 2:
        parent_folder = path_parts[-3] if len(path_parts) >= 3 else path_parts[-2]
        return f"{parent_folder}_best.pt"

    # Fallback ke nama file original
    return os.path.basename(model_path)


def analyze_model(model_path):

    try:
        if not os.path.exists(model_path):
            return {"error": f"File tidak ditemukan: {model_path}"}

        # Load model
        model = YOLO(model_path)
        pytorch_model = model.model
        pytorch_model.eval()

        # Input tensor dummy (1, 3, 640, 640)
        input_tensor = torch.randn(1, 3, 640, 640)

        # Hitung GFLOP dan parameter
        flops, params = profile(pytorch_model, inputs=(input_tensor,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

        # Ukuran file
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # Memory usage model
        model_size_mb = sum(p.numel() * p.element_size() for p in pytorch_model.parameters()) / (1024 * 1024)

        # Buat nama model yang lebih deskriptif dari path
        model_name = create_descriptive_name(model_path)

        return {
            "model_name": model_name,
            "full_path": model_path,
            "parameters": params,
            "parameters_formatted": params_formatted,
            "gflops": flops / 1e9,  # Convert to GFLOP
            "gflops_formatted": flops_formatted,
            "file_size_mb": file_size_mb,
            "memory_mb": model_size_mb,
            "task": model.task
        }

    except Exception as e:
        return {"error": f"Error analyzing {model_path}: {str(e)}"}


def compare_models(model_paths):

    print("YOLO Models Comparison")

    # Analisis semua model
    results = []
    for model_path in model_paths:
        print(f"Analyzing: {model_path}")
        result = analyze_model(model_path)
        if "error" not in result:
            results.append(result)
        else:
            print(f"Error: {result['error']}")

    if not results:
        print("Tidak ada model yang berhasil dianalisis")
        return

    # Buat DataFrame untuk comparison
    df = pd.DataFrame(results)

    # Sorting berdasarkan parameter (dari kecil ke besar)
    df = df.sort_values('parameters')

    print("\nCOMPARISON:")

    # Print hasil dalam format tabel
    print(f"{'Model Name':<20} {'Parameters':<12} {'GFLOP':<10} {'File Size (MB)':<15} {'Memory (MB)':<12}")

    for _, row in df.iterrows():
        print(
            f"{row['model_name']:<20} {row['parameters_formatted']:<12} {row['gflops']:<10.3f} {row['file_size_mb']:<15.2f} {row['memory_mb']:<12.2f}")


def main():

    model_paths = [
        "../yolo-gfl/runs/YOLO-GFL/weights/best.pt",
        "../yolov12/runs/YOLOv12/weights/best.pt",
    ]
    # Check dependencies
    try:
        import thop
        import ultralytics
        import pandas
        print("Dependencies: OK")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install thop ultralytics pandas")
        return

    # Jalankan comparison
    compare_models(model_paths)


if __name__ == "__main__":
    main()