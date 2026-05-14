# ============================================================
# KIỂM TRA CAMERA
# ============================================================
# Chạy thử:
# python src/check_camera.py --camera 0
# Nếu không hiện camera, thử --camera 1 hoặc --camera 2
# ============================================================

import argparse

import cv2


def main() -> None:
    bo_doc_tham_so = argparse.ArgumentParser()
    bo_doc_tham_so.add_argument("--camera", type=int, default=0, help="Chỉ số camera, thường là 0")
    tham_so = bo_doc_tham_so.parse_args()

    camera = cv2.VideoCapture(tham_so.camera)

    if not camera.isOpened():
        print("Không mở được camera. Hãy thử --camera 1 hoặc kiểm tra quyền camera.")
        return

    print("Camera đã mở. Nhấn q để thoát.")

    while True:
        thanh_cong, khung_hinh = camera.read()

        if not thanh_cong:
            print("Không đọc được khung hình từ camera.")
            break

        cv2.imshow("Kiem tra camera - nhan q de thoat", khung_hinh)

        phim = cv2.waitKey(1) & 0xFF
        if phim == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
