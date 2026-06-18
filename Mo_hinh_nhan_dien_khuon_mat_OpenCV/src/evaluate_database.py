# ============================================================
# ĐÁNH GIÁ NHANH DATABASE KHUÔN MẶT
# ============================================================
# Mục đích:
# - Kiểm tra dữ liệu khuôn mặt đã thu thập có ổn không.
# - Ước lượng độ chính xác bằng cách lấy từng ảnh mẫu ra kiểm tra lại.
# - Gợi ý ngưỡng nhận diện phù hợp.
#
# Chạy:
# python src/evaluate_database.py --threshold 0.50 --margin 0.06
# ============================================================

import argparse
from typing import Dict, List

import numpy as np

from common import chuan_hoa_vector, doc_database, nhan_dien_vector


def tao_vector_dai_dien_bo_mot_mau(
    danh_sach_ten: List[str],
    danh_sach_vector: np.ndarray,
    chi_so_bo_qua: int,
) -> Dict[str, np.ndarray]:
    ket_qua = {}
    tap_ten = sorted(set(danh_sach_ten))

    for ten in tap_ten:
        cac_vector = []

        for chi_so, ten_hien_tai in enumerate(danh_sach_ten):
            if chi_so == chi_so_bo_qua:
                continue

            if ten_hien_tai == ten:
                cac_vector.append(danh_sach_vector[chi_so])

        if len(cac_vector) > 0:
            vector_trung_binh = np.mean(np.array(cac_vector), axis=0)
            ket_qua[ten] = chuan_hoa_vector(vector_trung_binh)

    return ket_qua


def main() -> None:
    bo_doc_tham_so = argparse.ArgumentParser()
    bo_doc_tham_so.add_argument("--threshold", type=float, default=0.50, help="Ngưỡng nhận diện")
    bo_doc_tham_so.add_argument("--margin", type=float, default=0.06, help="Ngưỡng chênh lệch")
    tham_so = bo_doc_tham_so.parse_args()

    danh_sach_ten, danh_sach_vector, danh_sach_ten_mau = doc_database()

    so_mau = len(danh_sach_ten)
    so_dung = 0
    so_sai = 0
    so_unknown = 0

    print("Bắt đầu đánh giá nhanh database...")

    for chi_so in range(so_mau):
        ten_that = danh_sach_ten[chi_so]
        vector_can_kiem_tra = danh_sach_vector[chi_so]
        vector_dai_dien = tao_vector_dai_dien_bo_mot_mau(danh_sach_ten, danh_sach_vector, chi_so)

        ten_du_doan, diem, do_chenh = nhan_dien_vector(
            vector_can_kiem_tra,
            vector_dai_dien,
            nguong_nhan_dien=tham_so.threshold,
            nguong_chenh_lech=tham_so.margin,
        )

        if ten_du_doan == ten_that:
            so_dung += 1
        elif ten_du_doan == "Unknown":
            so_unknown += 1
        else:
            so_sai += 1
            print(
                f"Sai: ảnh={danh_sach_ten_mau[chi_so]}, "
                f"thật={ten_that}, dự đoán={ten_du_doan}, "
                f"score={diem:.4f}, margin={do_chenh:.4f}"
            )

    do_chinh_xac = so_dung / so_mau if so_mau > 0 else 0.0

    print("\nKẾT QUẢ ĐÁNH GIÁ")
    print(f"Tổng số mẫu: {so_mau}")
    print(f"Đúng: {so_dung}")
    print(f"Sai: {so_sai}")
    print(f"Không nhận diện: {so_unknown}")
    print(f"Độ chính xác ước lượng: {do_chinh_xac * 100:.2f}%")

    print("\nGợi ý chỉnh ngưỡng:")
    print("- Nếu nhận nhầm người: tăng --threshold lên 0.55 hoặc 0.60, tăng --margin lên 0.08.")
    print("- Nếu hay Unknown dù đúng người: giảm --threshold xuống 0.45, giữ ánh sáng tốt hơn.")


if __name__ == "__main__":
    main()
