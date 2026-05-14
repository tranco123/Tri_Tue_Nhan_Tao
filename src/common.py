# ============================================================
# FILE DÙNG CHUNG CHO ỨNG DỤNG ĐIỂM DANH KHUÔN MẶT
# ============================================================
# Vai trò:
# - Khai báo đường dẫn
# - Tạo bộ phát hiện mặt YuNet
# - Tạo bộ nhận diện mặt SFace
# - Trích xuất vector đặc trưng khuôn mặt
# - So khớp khuôn mặt
# - Ghi file điểm danh
# ============================================================

from __future__ import annotations

import csv
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------
# 1. KHAI BÁO ĐƯỜNG DẪN CHÍNH
# ------------------------------------------------------------
THU_MUC_GOC = Path(__file__).resolve().parents[1]
THU_MUC_DU_LIEU = THU_MUC_GOC / "data" / "faces"
THU_MUC_ONNX = THU_MUC_GOC / "models" / "onnx"
THU_MUC_DATABASE = THU_MUC_GOC / "models" / "database"
THU_MUC_DIEM_DANH = THU_MUC_GOC / "attendance"

FILE_MO_HINH_PHAT_HIEN = THU_MUC_ONNX / "face_detection_yunet_2023mar.onnx"
FILE_MO_HINH_NHAN_DIEN = THU_MUC_ONNX / "face_recognition_sface_2021dec.onnx"
FILE_DATABASE = THU_MUC_DATABASE / "face_database.npz"


# ------------------------------------------------------------
# 2. TẠO THƯ MỤC NẾU CHƯA CÓ
# ------------------------------------------------------------
def tao_thu_muc_can_thiet() -> None:
    THU_MUC_DU_LIEU.mkdir(parents=True, exist_ok=True)
    THU_MUC_ONNX.mkdir(parents=True, exist_ok=True)
    THU_MUC_DATABASE.mkdir(parents=True, exist_ok=True)
    THU_MUC_DIEM_DANH.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 3. CHUYỂN TÊN SINH VIÊN THÀNH TÊN THƯ MỤC AN TOÀN
# Ví dụ: "Trần Đức Cơ" -> "Tran_Duc_Co"
# ------------------------------------------------------------
def chuyen_ten_thanh_thu_muc(ho_ten: str) -> str:
    ten = ho_ten.strip()
    ten = unicodedata.normalize("NFD", ten)
    ten = "".join(ky_tu for ky_tu in ten if unicodedata.category(ky_tu) != "Mn")
    ten = re.sub(r"[^a-zA-Z0-9]+", "_", ten)
    ten = ten.strip("_")

    if ten == "":
        ten = "Sinh_Vien"

    return ten


# ------------------------------------------------------------
# 4. KIỂM TRA MÔ HÌNH ONNX ĐÃ CÓ CHƯA
# ------------------------------------------------------------
def kiem_tra_mo_hinh() -> None:
    if not FILE_MO_HINH_PHAT_HIEN.exists():
        raise FileNotFoundError(
            "Thiếu mô hình phát hiện khuôn mặt YuNet. "
            "Hãy chạy: python download_models.py"
        )

    if not FILE_MO_HINH_NHAN_DIEN.exists():
        raise FileNotFoundError(
            "Thiếu mô hình nhận diện khuôn mặt SFace. "
            "Hãy chạy: python download_models.py"
        )


# ------------------------------------------------------------
# 5. TẠO BỘ PHÁT HIỆN KHUÔN MẶT YUNET
# ------------------------------------------------------------
def tao_bo_phat_hien(
    chieu_rong: int,
    chieu_cao: int,
    nguong_phat_hien: float = 0.85,
    nguong_nms: float = 0.30,
    so_mat_toi_da: int = 5000,
):
    kiem_tra_mo_hinh()
    kich_thuoc = (chieu_rong, chieu_cao)

    # Một số phiên bản OpenCV dùng FaceDetectorYN.create,
    # một số phiên bản dùng FaceDetectorYN_create.
    if hasattr(cv2, "FaceDetectorYN"):
        bo_phat_hien = cv2.FaceDetectorYN.create(
            str(FILE_MO_HINH_PHAT_HIEN),
            "",
            kich_thuoc,
            nguong_phat_hien,
            nguong_nms,
            so_mat_toi_da,
        )
    else:
        bo_phat_hien = cv2.FaceDetectorYN_create(
            str(FILE_MO_HINH_PHAT_HIEN),
            "",
            kich_thuoc,
            nguong_phat_hien,
            nguong_nms,
            so_mat_toi_da,
        )

    return bo_phat_hien


# ------------------------------------------------------------
# 6. TẠO BỘ NHẬN DIỆN KHUÔN MẶT SFACE
# ------------------------------------------------------------
def tao_bo_nhan_dien():
    kiem_tra_mo_hinh()

    if hasattr(cv2, "FaceRecognizerSF"):
        bo_nhan_dien = cv2.FaceRecognizerSF.create(str(FILE_MO_HINH_NHAN_DIEN), "")
    else:
        bo_nhan_dien = cv2.FaceRecognizerSF_create(str(FILE_MO_HINH_NHAN_DIEN), "")

    return bo_nhan_dien


# ------------------------------------------------------------
# 7. PHÁT HIỆN CÁC KHUÔN MẶT TRONG ẢNH
# Kết quả mỗi khuôn mặt gồm:
# x, y, w, h, 5 điểm mốc khuôn mặt, điểm tin cậy
# ------------------------------------------------------------
def phat_hien_khuon_mat(bo_phat_hien, anh: np.ndarray) -> np.ndarray:
    chieu_cao, chieu_rong = anh.shape[:2]
    bo_phat_hien.setInputSize((chieu_rong, chieu_cao))
    ket_qua, danh_sach_mat = bo_phat_hien.detect(anh)

    if danh_sach_mat is None:
        return np.empty((0, 15), dtype=np.float32)

    return danh_sach_mat


# ------------------------------------------------------------
# 8. CHỌN KHUÔN MẶT LỚN NHẤT TRONG ẢNH
# Khi thu thập dữ liệu, ta thường chỉ muốn lấy mặt chính giữa/lớn nhất.
# ------------------------------------------------------------
def lay_khuon_mat_lon_nhat(danh_sach_mat: np.ndarray) -> Optional[np.ndarray]:
    if len(danh_sach_mat) == 0:
        return None

    chi_so_lon_nhat = 0
    dien_tich_lon_nhat = 0.0

    for chi_so, mat in enumerate(danh_sach_mat):
        rong = float(mat[2])
        cao = float(mat[3])
        dien_tich = rong * cao

        if dien_tich > dien_tich_lon_nhat:
            dien_tich_lon_nhat = dien_tich
            chi_so_lon_nhat = chi_so

    return danh_sach_mat[chi_so_lon_nhat]


# ------------------------------------------------------------
# 9. KIỂM TRA ẢNH CÓ ĐỦ NÉT KHÔNG
# Ảnh quá mờ sẽ làm giảm chất lượng nhận diện.
# ------------------------------------------------------------
def tinh_do_net(anh: np.ndarray) -> float:
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    do_net = cv2.Laplacian(anh_xam, cv2.CV_64F).var()
    return float(do_net)


# ------------------------------------------------------------
# 10. TRÍCH XUẤT VECTOR ĐẶC TRƯNG TỪ ẢNH GỐC VÀ VỊ TRÍ MẶT
# ------------------------------------------------------------
def trich_xuat_dac_trung_tu_anh_goc(
    bo_nhan_dien,
    anh_goc: np.ndarray,
    khuon_mat: np.ndarray,
) -> np.ndarray:
    anh_mat_can_chinh = bo_nhan_dien.alignCrop(anh_goc, khuon_mat)
    vector = bo_nhan_dien.feature(anh_mat_can_chinh)
    vector = np.array(vector).flatten().astype(np.float32)
    vector = chuan_hoa_vector(vector)
    return vector


# ------------------------------------------------------------
# 11. TRÍCH XUẤT VECTOR ĐẶC TRƯNG TỪ ẢNH MẶT ĐÃ CẮT/CĂN CHỈNH
# ------------------------------------------------------------
def trich_xuat_dac_trung_tu_anh_mat(
    bo_nhan_dien,
    anh_mat: np.ndarray,
) -> np.ndarray:
    anh_mat = cv2.resize(anh_mat, (112, 112))
    vector = bo_nhan_dien.feature(anh_mat)
    vector = np.array(vector).flatten().astype(np.float32)
    vector = chuan_hoa_vector(vector)
    return vector


# ------------------------------------------------------------
# 12. CHUẨN HÓA VECTOR ĐỂ SO KHỚP COSINE ỔN ĐỊNH HƠN
# ------------------------------------------------------------
def chuan_hoa_vector(vector: np.ndarray) -> np.ndarray:
    do_dai = np.linalg.norm(vector)

    if do_dai == 0:
        return vector

    return vector / do_dai


# ------------------------------------------------------------
# 13. TÍNH ĐỘ TƯƠNG ĐỒNG COSINE
# Giá trị càng cao thì hai khuôn mặt càng giống nhau.
# ------------------------------------------------------------
def tinh_do_tuong_dong(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    vector_1 = chuan_hoa_vector(vector_1)
    vector_2 = chuan_hoa_vector(vector_2)
    diem = float(np.dot(vector_1, vector_2))
    return diem


# ------------------------------------------------------------
# 14. VẼ KHUNG KHUÔN MẶT VÀ TÊN LÊN ẢNH
# ------------------------------------------------------------
def ve_khung_khuon_mat(
    anh: np.ndarray,
    khuon_mat: np.ndarray,
    noi_dung: str,
    mau: Tuple[int, int, int],
) -> None:
    x = int(khuon_mat[0])
    y = int(khuon_mat[1])
    w = int(khuon_mat[2])
    h = int(khuon_mat[3])

    cv2.rectangle(anh, (x, y), (x + w, y + h), mau, 2)
    cv2.rectangle(anh, (x, y - 32), (x + w, y), mau, -1)
    cv2.putText(
        anh,
        noi_dung,
        (x + 5, y - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ------------------------------------------------------------
# 15. LƯU DATABASE KHUÔN MẶT
# ------------------------------------------------------------
def luu_database(
    danh_sach_ten: List[str],
    danh_sach_vector: List[np.ndarray],
    danh_sach_ten_mau: List[str],
) -> None:
    THU_MUC_DATABASE.mkdir(parents=True, exist_ok=True)

    mang_ten = np.array(danh_sach_ten, dtype=object)
    mang_vector = np.array(danh_sach_vector, dtype=np.float32)
    mang_ten_mau = np.array(danh_sach_ten_mau, dtype=object)

    np.savez_compressed(
        FILE_DATABASE,
        ten=mang_ten,
        vector=mang_vector,
        ten_mau=mang_ten_mau,
    )


# ------------------------------------------------------------
# 16. ĐỌC DATABASE KHUÔN MẶT
# ------------------------------------------------------------
def doc_database() -> Tuple[List[str], np.ndarray, List[str]]:
    if not FILE_DATABASE.exists():
        raise FileNotFoundError(
            "Chưa có database khuôn mặt. "
            "Hãy chạy: python src/build_database.py"
        )

    du_lieu = np.load(FILE_DATABASE, allow_pickle=True)
    danh_sach_ten = du_lieu["ten"].tolist()
    danh_sach_vector = du_lieu["vector"].astype(np.float32)
    danh_sach_ten_mau = du_lieu["ten_mau"].tolist()

    return danh_sach_ten, danh_sach_vector, danh_sach_ten_mau


# ------------------------------------------------------------
# 17. TẠO VECTOR ĐẠI DIỆN CHO TỪNG NGƯỜI
# Mỗi người có nhiều ảnh, ta lấy trung bình vector để ổn định hơn.
# ------------------------------------------------------------
def tao_vector_dai_dien_theo_nguoi(
    danh_sach_ten: List[str],
    danh_sach_vector: np.ndarray,
) -> Dict[str, np.ndarray]:
    ket_qua: Dict[str, np.ndarray] = {}
    tap_ten = sorted(set(danh_sach_ten))

    for ten in tap_ten:
        cac_vector = []

        for chi_so, ten_hien_tai in enumerate(danh_sach_ten):
            if ten_hien_tai == ten:
                cac_vector.append(danh_sach_vector[chi_so])

        if len(cac_vector) == 0:
            continue

        vector_trung_binh = np.mean(np.array(cac_vector), axis=0)
        ket_qua[ten] = chuan_hoa_vector(vector_trung_binh)

    return ket_qua


# ------------------------------------------------------------
# 18. NHẬN DIỆN 1 VECTOR KHUÔN MẶT
# Điều kiện nhận diện gồm:
# - Điểm giống tốt nhất >= ngưỡng
# - Khoảng cách giữa người tốt nhất và người thứ hai đủ lớn
# ------------------------------------------------------------
def nhan_dien_vector(
    vector_can_nhan_dien: np.ndarray,
    vector_dai_dien: Dict[str, np.ndarray],
    nguong_nhan_dien: float = 0.50,
    nguong_chenh_lech: float = 0.06,
) -> Tuple[str, float, float]:
    danh_sach_diem = []

    for ten, vector_nguoi in vector_dai_dien.items():
        diem = tinh_do_tuong_dong(vector_can_nhan_dien, vector_nguoi)
        danh_sach_diem.append((ten, diem))

    if len(danh_sach_diem) == 0:
        return "Unknown", 0.0, 0.0

    danh_sach_diem.sort(key=lambda phan_tu: phan_tu[1], reverse=True)

    ten_tot_nhat = danh_sach_diem[0][0]
    diem_tot_nhat = float(danh_sach_diem[0][1])

    if len(danh_sach_diem) >= 2:
        diem_thu_hai = float(danh_sach_diem[1][1])
    else:
        diem_thu_hai = 0.0

    do_chenh_lech = diem_tot_nhat - diem_thu_hai

    if diem_tot_nhat >= nguong_nhan_dien and do_chenh_lech >= nguong_chenh_lech:
        return ten_tot_nhat, diem_tot_nhat, do_chenh_lech

    return "Unknown", diem_tot_nhat, do_chenh_lech


# ------------------------------------------------------------
# 19. GHI ĐIỂM DANH RA FILE CSV THEO NGÀY
# ------------------------------------------------------------
def ghi_diem_danh(ho_ten: str, diem_tin_cay: float) -> bool:
    THU_MUC_DIEM_DANH.mkdir(parents=True, exist_ok=True)
    bay_gio = datetime.now()
    ngay = bay_gio.strftime("%Y-%m-%d")
    thoi_gian = bay_gio.strftime("%H:%M:%S")
    file_diem_danh = THU_MUC_DIEM_DANH / f"diem_danh_{ngay}.csv"

    da_ton_tai = file_diem_danh.exists()

    # Nếu sinh viên đã được điểm danh trong ngày thì không ghi trùng.
    if da_ton_tai:
        with open(file_diem_danh, "r", encoding="utf-8-sig", newline="") as tep:
            bo_doc = csv.DictReader(tep)
            for dong in bo_doc:
                if dong.get("ho_ten") == ho_ten:
                    return False

    with open(file_diem_danh, "a", encoding="utf-8-sig", newline="") as tep:
        ten_cot = ["ho_ten", "ngay", "thoi_gian", "diem_tin_cay"]
        bo_ghi = csv.DictWriter(tep, fieldnames=ten_cot)

        if not da_ton_tai:
            bo_ghi.writeheader()

        bo_ghi.writerow(
            {
                "ho_ten": ho_ten,
                "ngay": ngay,
                "thoi_gian": thoi_gian,
                "diem_tin_cay": f"{diem_tin_cay:.4f}",
            }
        )

    return True
