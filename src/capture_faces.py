# ============================================================
# THU THẬP ẢNH KHUÔN MẶT SINH VIÊN
# ============================================================
# Ví dụ chạy:
# python src/capture_faces.py --person_name "Tran Duc Co" --num_samples 50
#
# Cách dùng:
# - Camera sẽ mở lên.
# - Đưa mặt vào khung hình, nhìn thẳng, đủ sáng.
# - Ảnh sẽ tự lưu khi khuôn mặt đủ lớn và đủ nét.
# - Nhấn q để thoát sớm.
# ============================================================

import argparse
from datetime import datetime

import cv2

from common import (
    THU_MUC_DU_LIEU,
    chuyen_ten_thanh_thu_muc,
    lay_khuon_mat_lon_nhat,
    phat_hien_khuon_mat,
    tao_bo_nhan_dien,
    tao_bo_phat_hien,
    tao_thu_muc_can_thiet,
    tinh_do_net,
)


def main() -> None:
    bo_doc_tham_so = argparse.ArgumentParser()
    bo_doc_tham_so.add_argument("--person_name", required=True, help="Họ tên sinh viên")
    bo_doc_tham_so.add_argument("--camera", type=int, default=0, help="Chỉ số camera")
    bo_doc_tham_so.add_argument("--num_samples", type=int, default=50, help="Số ảnh cần thu thập")
    bo_doc_tham_so.add_argument("--min_face_size", type=int, default=90, help="Kích thước mặt tối thiểu")
    bo_doc_tham_so.add_argument("--min_sharpness", type=float, default=45.0, help="Độ nét tối thiểu")
    bo_doc_tham_so.add_argument("--delay", type=int, default=4, help="Cứ bao nhiêu khung hình thì lưu 1 ảnh")
    tham_so = bo_doc_tham_so.parse_args()

    tao_thu_muc_can_thiet()

    ten_thu_muc = chuyen_ten_thanh_thu_muc(tham_so.person_name)
    thu_muc_nguoi = THU_MUC_DU_LIEU / ten_thu_muc
    thu_muc_nguoi.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(tham_so.camera)

    if not camera.isOpened():
        print("Không mở được camera. Hãy thử --camera 1 hoặc kiểm tra quyền camera.")
        return

    chieu_rong = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    chieu_cao = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bo_phat_hien = tao_bo_phat_hien(chieu_rong, chieu_cao, nguong_phat_hien=0.85)
    bo_nhan_dien = tao_bo_nhan_dien()

    so_anh_da_luu = 0
    dem_khung_hinh = 0

    print("Bắt đầu thu thập dữ liệu khuôn mặt.")
    print("Gợi ý: xoay mặt nhẹ trái/phải, lên/xuống, thay đổi biểu cảm một chút.")
    print("Nhấn q để thoát.")

    while so_anh_da_luu < tham_so.num_samples:
        thanh_cong, khung_hinh = camera.read()

        if not thanh_cong:
            print("Không đọc được khung hình từ camera.")
            break

        dem_khung_hinh += 1
        danh_sach_mat = phat_hien_khuon_mat(bo_phat_hien, khung_hinh)
        mat_chinh = lay_khuon_mat_lon_nhat(danh_sach_mat)

        thong_bao = "Khong thay mat"
        mau = (0, 0, 255)

        if mat_chinh is not None:
            x = int(mat_chinh[0])
            y = int(mat_chinh[1])
            w = int(mat_chinh[2])
            h = int(mat_chinh[3])

            vung_mat = khung_hinh[max(y, 0): max(y + h, 0), max(x, 0): max(x + w, 0)]
            do_net = tinh_do_net(vung_mat) if vung_mat.size > 0 else 0.0

            du_lon = w >= tham_so.min_face_size and h >= tham_so.min_face_size
            du_net = do_net >= tham_so.min_sharpness

            if du_lon and du_net:
                thong_bao = f"Tot - dang luu {so_anh_da_luu}/{tham_so.num_samples}"
                mau = (0, 180, 0)

                if dem_khung_hinh % tham_so.delay == 0:
                    anh_mat_can_chinh = bo_nhan_dien.alignCrop(khung_hinh, mat_chinh)
                    thoi_gian = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    ten_file = thu_muc_nguoi / f"{ten_thu_muc}_{thoi_gian}.jpg"
                    cv2.imwrite(str(ten_file), anh_mat_can_chinh)
                    so_anh_da_luu += 1
            else:
                thong_bao = f"Mat nho/mo - net={do_net:.1f}"
                mau = (0, 165, 255)

            cv2.rectangle(khung_hinh, (x, y), (x + w, y + h), mau, 2)

        cv2.putText(
            khung_hinh,
            thong_bao,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            mau,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            khung_hinh,
            "Nhan q de thoat",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Thu thap khuon mat", khung_hinh)

        phim = cv2.waitKey(1) & 0xFF
        if phim == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print(f"Đã lưu {so_anh_da_luu} ảnh vào thư mục: {thu_muc_nguoi}")
    print("Tiếp theo hãy chạy: python src/build_database.py")


if __name__ == "__main__":
    main()
