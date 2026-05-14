# ============================================================
# TẢI MÔ HÌNH ONNX CHO ỨNG DỤNG ĐIỂM DANH KHUÔN MẶT
# ============================================================
# Chạy file này một lần để tải 2 mô hình:
# 1. YuNet: phát hiện khuôn mặt
# 2. SFace: trích xuất đặc trưng / nhận diện khuôn mặt
# ============================================================

from pathlib import Path
from urllib.request import urlretrieve

THU_MUC_GOC = Path(__file__).resolve().parent
THU_MUC_ONNX = THU_MUC_GOC / "models" / "onnx"
THU_MUC_ONNX.mkdir(parents=True, exist_ok=True)

DANH_SACH_MO_HINH = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def tai_mo_hinh(ten_file: str, duong_dan_tai: str) -> None:
    duong_dan_luu = THU_MUC_ONNX / ten_file

    if duong_dan_luu.exists() and duong_dan_luu.stat().st_size > 1000:
        print(f"Đã có: {duong_dan_luu}")
        return

    print(f"Đang tải: {ten_file}")
    print("Vui lòng chờ, file nhận diện có thể hơi nặng...")
    urlretrieve(duong_dan_tai, duong_dan_luu)
    print(f"Đã tải xong: {duong_dan_luu}")


def main() -> None:
    for ten_file, duong_dan_tai in DANH_SACH_MO_HINH.items():
        tai_mo_hinh(ten_file, duong_dan_tai)

    print("\nHoàn tất tải mô hình.")


if __name__ == "__main__":
    main()
