# ============================================================
# NHẬN DIỆN KHUÔN MẶT VÀ ĐIỂM DANH BẰNG WEBCAM
# ============================================================
# Chạy:
# python src/recognize_attendance.py
#
# Nếu nhận sai nhiều: tăng --threshold, ví dụ 0.55 hoặc 0.60
# Nếu khó nhận đúng: giảm nhẹ --threshold, ví dụ 0.45
# ============================================================

import argparse

import cv2

from common import (
    doc_database,
    ghi_diem_danh,
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
    bo_doc_tham_so.add_argument("--camera", type=int, default=0, help="Chỉ số camera")
    bo_doc_tham_so.add_argument("--threshold", type=float, default=0.50, help="Ngưỡng nhận diện")
    bo_doc_tham_so.add_argument("--margin", type=float, default=0.06, help="Ngưỡng chênh lệch với người thứ hai")
    bo_doc_tham_so.add_argument("--det_score", type=float, default=0.85, help="Ngưỡng phát hiện khuôn mặt")
    tham_so = bo_doc_tham_so.parse_args()

    danh_sach_ten, danh_sach_vector, danh_sach_ten_mau = doc_database()
    vector_dai_dien = tao_vector_dai_dien_theo_nguoi(danh_sach_ten, danh_sach_vector)

    print("Đã đọc database khuôn mặt.")
    print(f"Số người trong database: {len(vector_dai_dien)}")
    print(f"Tổng số ảnh mẫu: {len(danh_sach_ten_mau)}")
    print("Nhấn q để thoát.")

    camera = cv2.VideoCapture(tham_so.camera)

    if not camera.isOpened():
        print("Không mở được camera. Hãy thử --camera 1 hoặc kiểm tra quyền camera.")
        return

    chieu_rong = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    chieu_cao = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bo_phat_hien = tao_bo_phat_hien(
        chieu_rong,
        chieu_cao,
        nguong_phat_hien=tham_so.det_score,
    )
    bo_nhan_dien = tao_bo_nhan_dien()

    while True:
        thanh_cong, khung_hinh = camera.read()

        if not thanh_cong:
            print("Không đọc được khung hình từ camera.")
            break

        danh_sach_mat = phat_hien_khuon_mat(bo_phat_hien, khung_hinh)

        for mat in danh_sach_mat:
            try:
                vector = trich_xuat_dac_trung_tu_anh_goc(bo_nhan_dien, khung_hinh, mat)
                ten, diem, do_chenh = nhan_dien_vector(
                    vector,
                    vector_dai_dien,
                    nguong_nhan_dien=tham_so.threshold,
                    nguong_chenh_lech=tham_so.margin,
                )
            except Exception:
                ten = "Unknown"
                diem = 0.0
                do_chenh = 0.0

            if ten != "Unknown":
                da_ghi = ghi_diem_danh(ten, diem)

                if da_ghi:
                    noi_dung = f"{ten} - Diem danh"
                else:
                    noi_dung = f"{ten} - Da co"

                mau = (0, 180, 0)
            else:
                noi_dung = f"Unknown {diem:.2f}"
                mau = (0, 0, 255)

            # Đưa thêm điểm và độ chênh vào terminal để dễ tinh chỉnh ngưỡng.
            print(f"Nhan dien: {ten}, score={diem:.4f}, margin={do_chenh:.4f}", end="\r")
            ve_khung_khuon_mat(khung_hinh, mat, noi_dung, mau)

        cv2.putText(
            khung_hinh,
            "Nhan q de thoat",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Nhan dien khuon mat diem danh", khung_hinh)

        phim = cv2.waitKey(1) & 0xFF
        if phim == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("\nĐã thoát chương trình.")


if __name__ == "__main__":
    main()
