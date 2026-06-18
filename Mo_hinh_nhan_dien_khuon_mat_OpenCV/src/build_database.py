# ============================================================
# XÂY DỰNG DATABASE KHUÔN MẶT
# ============================================================
# File này đọc các ảnh đã thu thập trong data/faces,
# trích xuất vector đặc trưng bằng SFace,
# sau đó lưu vào models/database/face_database.npz
#
# Chạy:
# python src/build_database.py
# ============================================================

from pathlib import Path

import cv2

from common import (
    THU_MUC_DU_LIEU,
    FILE_DATABASE,
    luu_database,
    tao_bo_nhan_dien,
    tao_thu_muc_can_thiet,
    trich_xuat_dac_trung_tu_anh_mat,
)

DINH_DANG_ANH = ["*.jpg", "*.jpeg", "*.png"]


def lay_danh_sach_anh(thu_muc_nguoi: Path):
    danh_sach_anh = []

    for dinh_dang in DINH_DANG_ANH:
        for duong_dan_anh in thu_muc_nguoi.glob(dinh_dang):
            danh_sach_anh.append(duong_dan_anh)

    danh_sach_anh.sort()
    return danh_sach_anh


def main() -> None:
    tao_thu_muc_can_thiet()
    bo_nhan_dien = tao_bo_nhan_dien()

    danh_sach_ten = []
    danh_sach_vector = []
    danh_sach_ten_mau = []

    if not THU_MUC_DU_LIEU.exists():
        print("Chưa có thư mục dữ liệu.")
        return

    danh_sach_thu_muc_nguoi = []

    for thu_muc in THU_MUC_DU_LIEU.iterdir():
        if thu_muc.is_dir():
            danh_sach_thu_muc_nguoi.append(thu_muc)

    danh_sach_thu_muc_nguoi.sort()

    if len(danh_sach_thu_muc_nguoi) == 0:
        print("Chưa có dữ liệu khuôn mặt.")
        print("Hãy chạy ví dụ: python src/capture_faces.py --person_name \"Tran Duc Co\"")
        return

    print("Bắt đầu xây dựng database khuôn mặt...")

    for thu_muc_nguoi in danh_sach_thu_muc_nguoi:
        ten_nguoi = thu_muc_nguoi.name
        danh_sach_anh = lay_danh_sach_anh(thu_muc_nguoi)

        if len(danh_sach_anh) == 0:
            print(f"Bỏ qua {ten_nguoi}: không có ảnh.")
            continue

        so_anh_hop_le = 0

        for duong_dan_anh in danh_sach_anh:
            anh = cv2.imread(str(duong_dan_anh))

            if anh is None:
                print(f"Không đọc được ảnh: {duong_dan_anh}")
                continue

            try:
                vector = trich_xuat_dac_trung_tu_anh_mat(bo_nhan_dien, anh)
            except Exception as loi:
                print(f"Lỗi khi xử lý ảnh {duong_dan_anh.name}: {loi}")
                continue

            danh_sach_ten.append(ten_nguoi)
            danh_sach_vector.append(vector)
            danh_sach_ten_mau.append(duong_dan_anh.name)
            so_anh_hop_le += 1

        print(f"{ten_nguoi}: {so_anh_hop_le}/{len(danh_sach_anh)} ảnh hợp lệ")

    if len(danh_sach_vector) == 0:
        print("Không tạo được database vì không có ảnh hợp lệ.")
        return

    luu_database(danh_sach_ten, danh_sach_vector, danh_sach_ten_mau)

    print("\nHoàn tất.")
    print(f"Đã lưu database tại: {FILE_DATABASE}")
    print(f"Tổng số vector khuôn mặt: {len(danh_sach_vector)}")
    print("Tiếp theo hãy chạy: python src/recognize_attendance.py")


if __name__ == "__main__":
    main()
