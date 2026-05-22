from pathlib import Path
import pandas as pd


# Duong dan thu muc chua du lieu khuon mat
thu_muc_faces = Path("data/faces")

# Duong dan thu muc luu ket qua
thu_muc_ket_qua = Path("attendance")
thu_muc_ket_qua.mkdir(parents=True, exist_ok=True)

# Cac dinh dang anh duoc chap nhan
dinh_dang_anh = ["*.jpg", "*.jpeg", "*.png"]

# Danh sach luu ket qua thong ke
du_lieu_thong_ke = []

# Kiem tra thu muc data/faces co ton tai khong
if not thu_muc_faces.exists():
    print("Khong tim thay thu muc data/faces")
    exit()

# Lay danh sach thu muc sinh vien
danh_sach_thu_muc = []

for thu_muc_sinh_vien in thu_muc_faces.iterdir():
    if thu_muc_sinh_vien.is_dir():
        danh_sach_thu_muc.append(thu_muc_sinh_vien)

# Sap xep ten sinh vien theo alphabet cho bang dep hon
danh_sach_thu_muc = sorted(danh_sach_thu_muc, key=lambda x: x.name.lower())

# Duyet tung thu muc sinh vien
for stt, thu_muc_sinh_vien in enumerate(danh_sach_thu_muc, start=1):
    ten_sinh_vien = thu_muc_sinh_vien.name

    danh_sach_anh = []

    for dinh_dang in dinh_dang_anh:
        danh_sach_anh.extend(list(thu_muc_sinh_vien.glob(dinh_dang)))

    so_anh_thu_thap = len(danh_sach_anh)

    if so_anh_thu_thap >= 40:
        ghi_chu = "Dat yeu cau"
    elif so_anh_thu_thap > 0:
        ghi_chu = "Can bo sung them anh"
    else:
        ghi_chu = "Chua co du lieu"

    du_lieu_thong_ke.append({
        "STT": stt,
        "Ho ten sinh vien": ten_sinh_vien,
        "So anh khuon mat": so_anh_thu_thap,
        "Thu muc luu tru": str(thu_muc_sinh_vien),
        "Ghi chu": ghi_chu
    })

# Tao DataFrame
bang_thong_ke = pd.DataFrame(du_lieu_thong_ke)

# Tinh tong so sinh vien va tong so anh
tong_so_sinh_vien = len(bang_thong_ke)
tong_so_anh = bang_thong_ke["So anh khuon mat"].sum()

# In bang ra man hinh
print("\nBANG 5.1. THONG KE DU LIEU KHUON MAT THUC NGHIEM\n")
print(bang_thong_ke.to_string(index=False))

print("\nTHONG TIN TONG HOP")
print(f"Tong so sinh vien: {tong_so_sinh_vien}")
print(f"Tong so anh khuon mat: {tong_so_anh}")

