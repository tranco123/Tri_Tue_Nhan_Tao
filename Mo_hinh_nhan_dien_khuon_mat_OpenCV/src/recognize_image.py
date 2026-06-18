# ============================================================
# NHẬN DIỆN KHUÔN MẶT TRÊN ẢNH TĨNH
# ============================================================
# Dùng để demo nếu trên lớp không dùng được webcam.
#
# Chạy:
# python src/recognize_image.py --image duong_dan_anh.jpg
# ============================================================

import argparse
from pathlib import Path

import cv2

from common import (
    doc_database,
    nhan_dien_vector,
    phat_hien_khuon_mat,
    tao_bo_nhan_dien,
    tao_bo_phat_hien,
    tao_vector_dai_dien_theo_nguoi,
    trich_xuat_dac_trung_tu_anh_goc,
    ve_khung_khuon_mat,
)


def main() -> None:
    bo_doc_tham_so = argparse.ArgumentParser()
    bo_doc_tham_so.add_argument("--image", required=True, help="Đường dẫn ảnh cần nhận diện")
    bo_doc_tham_so.add_argument("--threshold", type=float, default=0.50, help="Ngưỡng nhận diện")
    bo_doc_tham_so.add_argument("--margin", type=float, default=0.06, help="Ngưỡng chênh lệch")
    tham_so = bo_doc_tham_so.parse_args()

    duong_dan_anh = Path(tham_so.image)
    anh = cv2.imread(str(duong_dan_anh))

    if anh is None:
        print(f"Không đọc được ảnh: {duong_dan_anh}")
        return

    danh_sach_ten, danh_sach_vector, danh_sach_ten_mau = doc_database()
    vector_dai_dien = tao_vector_dai_dien_theo_nguoi(danh_sach_ten, danh_sach_vector)

    chieu_cao, chieu_rong = anh.shape[:2]
    bo_phat_hien = tao_bo_phat_hien(chieu_rong, chieu_cao, nguong_phat_hien=0.85)
    bo_nhan_dien = tao_bo_nhan_dien()

    danh_sach_mat = phat_hien_khuon_mat(bo_phat_hien, anh)

    if len(danh_sach_mat) == 0:
        print("Không phát hiện được khuôn mặt trong ảnh.")
        return

    for mat in danh_sach_mat:
        vector = trich_xuat_dac_trung_tu_anh_goc(bo_nhan_dien, anh, mat)
        ten, diem, do_chenh = nhan_dien_vector(
            vector,
            vector_dai_dien,
            nguong_nhan_dien=tham_so.threshold,
            nguong_chenh_lech=tham_so.margin,
        )

        if ten == "Unknown":
            noi_dung = f"Unknown {diem:.2f}"
            mau = (0, 0, 255)
        else:
            noi_dung = f"{ten} {diem:.2f}"
            mau = (0, 180, 0)

        ve_khung_khuon_mat(anh, mat, noi_dung, mau)
        print(f"Kết quả: {ten}, score={diem:.4f}, margin={do_chenh:.4f}")

    duong_dan_luu = duong_dan_anh.with_name(duong_dan_anh.stem + "_ket_qua.jpg")
    cv2.imwrite(str(duong_dan_luu), anh)
    print(f"Đã lưu ảnh kết quả: {duong_dan_luu}")

    cv2.imshow("Ket qua nhan dien", anh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
