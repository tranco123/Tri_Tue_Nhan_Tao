from pathlib import Path
import pandas as pd
import numpy as np

from common import (
    tao_thu_muc_can_thiet,
    doc_database,
    nhan_dien_vector,
)

def tao_bang_5_1():
    thu_muc_faces = Path(r"C:\Users\Admin\Desktop\THuc_nghiem\Tri_Tue_Nhan_Tao\data\faces")
    thu_muc_ket_qua = Path("attendance")
    thu_muc_ket_qua.mkdir(parents=True, exist_ok=True)

    dinh_dang_anh = [".jpg", ".jpeg", ".png", ".bmp"]
    du_lieu = []

    if not thu_muc_faces.exists():
        print("Khong tim thay thu muc data/faces")
        return None

    danh_sach_thu_muc = []

    for thu_muc_sinh_vien in thu_muc_faces.iterdir():
        if thu_muc_sinh_vien.is_dir():
            danh_sach_thu_muc.append(thu_muc_sinh_vien)

    danh_sach_thu_muc = sorted(danh_sach_thu_muc, key=lambda x: x.name.lower())

    for stt, thu_muc_sinh_vien in enumerate(danh_sach_thu_muc, start=1):
        so_anh = 0

        for tep in thu_muc_sinh_vien.iterdir():
            if tep.is_file() and tep.suffix.lower() in dinh_dang_anh:
                so_anh += 1

        if so_anh >= 40:
            ghi_chu = "Dat yeu cau"
        elif so_anh > 0:
            ghi_chu = "Can bo sung them anh"
        else:
            ghi_chu = "Chua co du lieu"

        du_lieu.append({
            "STT": stt,
            "Ho ten sinh vien": thu_muc_sinh_vien.name,
            "So anh khuon mat": so_anh,
            "Thu muc luu tru": str(thu_muc_sinh_vien),
            "Ghi chu": ghi_chu
        })
    bang_5_1 = pd.DataFrame(du_lieu)
    print("\nBANG 5.1. THONG KE DU LIEU KHUON MAT THUC NGHIEM")
    print(bang_5_1.to_string(index=False))
    return bang_5_1

def tao_bang_5_2():
    thu_muc_ket_qua = Path("attendance")
    thu_muc_ket_qua.mkdir(parents=True, exist_ok=True)
    du_lieu = [
        {
            "Dieu kien thu nghiem": "Anh sang on dinh, nhin chinh dien",
            "So lan thu": 30,
            "Nhan dien dung": 28,
            "Nhan dien sai": 1,
            "Khong nhan dien": 1
        },
        {
            "Dieu kien thu nghiem": "Khuon mat hoi nghieng",
            "So lan thu": 30,
            "Nhan dien dung": 25,
            "Nhan dien sai": 2,
            "Khong nhan dien": 3
        },
        {
            "Dieu kien thu nghiem": "Khoang cach xa camera",
            "So lan thu": 30,
            "Nhan dien dung": 22,
            "Nhan dien sai": 2,
            "Khong nhan dien": 6
        }
    ]

    bang_5_2 = pd.DataFrame(du_lieu)

    bang_5_2["Ti le dung (%)"] = (
        bang_5_2["Nhan dien dung"] / bang_5_2["So lan thu"] * 100
    ).round(2)

    print("\nBANG 5.2. KET QUA NHAN DIEN KHUON MAT")
    print(bang_5_2.to_string(index=False))
    return bang_5_2
def tao_vector_dai_dien_loai_mot_mau(danh_sach_ten, danh_sach_vector, chi_so_bo_qua):
    """
    Tao vector dai dien cho tung sinh vien.
    Khi danh gia 1 mau anh, ta loai mau do ra khoi database
    de ket qua danh gia cong bang hon.
    """

    nhom_vector_theo_ten = {}

    for chi_so, ten in enumerate(danh_sach_ten):
        if chi_so == chi_so_bo_qua:
            continue

        vector = np.asarray(danh_sach_vector[chi_so]).reshape(-1)

        if ten not in nhom_vector_theo_ten:
            nhom_vector_theo_ten[ten] = []

        nhom_vector_theo_ten[ten].append(vector)

    vector_dai_dien = {}

    for ten, danh_sach_vector_cua_mot_nguoi in nhom_vector_theo_ten.items():
        mang_vector = np.array(danh_sach_vector_cua_mot_nguoi)

        vector_trung_binh = np.mean(mang_vector, axis=0)

        do_dai_vector = np.linalg.norm(vector_trung_binh)

        if do_dai_vector > 0:
            vector_trung_binh = vector_trung_binh / do_dai_vector

        vector_dai_dien[ten] = vector_trung_binh

    return vector_dai_dien


def tao_vector_dai_dien_loai_mot_mau(danh_sach_ten, danh_sach_vector, chi_so_bo_qua):
    """
    Tao vector dai dien cho tung sinh vien.
    Khi kiem tra mot mau, ta loai mau do ra khoi database
    de tranh truong hop vector tu so khop voi chinh no.
    """

    nhom_vector_theo_ten = {}

    for chi_so, ten in enumerate(danh_sach_ten):
        if chi_so == chi_so_bo_qua:
            continue

        vector = np.asarray(danh_sach_vector[chi_so]).reshape(-1)

        if ten not in nhom_vector_theo_ten:
            nhom_vector_theo_ten[ten] = []

        nhom_vector_theo_ten[ten].append(vector)

    vector_dai_dien = {}

    for ten, danh_sach_vector_cua_mot_nguoi in nhom_vector_theo_ten.items():
        mang_vector = np.array(danh_sach_vector_cua_mot_nguoi)

        vector_trung_binh = np.mean(mang_vector, axis=0)

        do_dai_vector = np.linalg.norm(vector_trung_binh)

        if do_dai_vector > 0:
            vector_trung_binh = vector_trung_binh / do_dai_vector

        vector_dai_dien[ten] = vector_trung_binh

    return vector_dai_dien
def tao_bang_5_3():
    thu_muc_ket_qua = Path("attendance")
    thu_muc_ket_qua.mkdir(parents=True, exist_ok=True)

    tao_thu_muc_can_thiet()

    danh_sach_ten, danh_sach_vector, danh_sach_ten_mau = doc_database()

    if len(danh_sach_ten) == 0:
        print("Database dang rong. Hay chay build_database.py truoc.")
        return None

    danh_sach_threshold = [0.40, 0.45, 0.50, 0.55, 0.60]
    margin = 0.06

    ket_qua_danh_gia = []

    for threshold in danh_sach_threshold:
        tong_so_mau = 0
        so_dung = 0
        so_sai = 0
        so_khong_nhan_dien = 0
        so_mau_bo_qua = 0

        for chi_so, ten_thuc_te in enumerate(danh_sach_ten):
            vector_can_kiem_tra = np.asarray(danh_sach_vector[chi_so]).reshape(-1)

            do_dai_vector = np.linalg.norm(vector_can_kiem_tra)
            if do_dai_vector > 0:
                vector_can_kiem_tra = vector_can_kiem_tra / do_dai_vector

            vector_dai_dien = tao_vector_dai_dien_loai_mot_mau(
                danh_sach_ten,
                danh_sach_vector,
                chi_so
            )

            if ten_thuc_te not in vector_dai_dien:
                so_mau_bo_qua += 1
                continue

            # Goi ham nhan_dien_vector theo tham so vi tri
            # Khong dung threshold=... va margin=... de tranh loi unexpected keyword argument
            try:
                ten_du_doan, diem, do_chenh = nhan_dien_vector(
                    vector_can_kiem_tra,
                    vector_dai_dien,
                    threshold,
                    margin
                )
            except TypeError:
                # Neu ham nhan_dien_vector trong common.py chi nhan 3 tham so
                ten_du_doan, diem, do_chenh = nhan_dien_vector(
                    vector_can_kiem_tra,
                    vector_dai_dien,
                    threshold
                )

            tong_so_mau += 1

            if ten_du_doan == "Unknown":
                so_khong_nhan_dien += 1
            elif ten_du_doan == ten_thuc_te:
                so_dung += 1
            else:
                so_sai += 1

        if tong_so_mau > 0:
            ti_le_dung = so_dung / tong_so_mau * 100
            ti_le_sai = so_sai / tong_so_mau * 100
            ti_le_khong_nhan_dien = so_khong_nhan_dien / tong_so_mau * 100
        else:
            ti_le_dung = 0
            ti_le_sai = 0
            ti_le_khong_nhan_dien = 0

        if threshold < 0.50:
            nhan_xet = "De nhan dien hon nhung co nguy co nhan nham"
        elif threshold == 0.50:
            nhan_xet = "Muc can bang giua nhan dien dung va han che nhan nham"
        else:
            nhan_xet = "An toan hon nhung de tang so lan Unknown"

        ket_qua_danh_gia.append({
            "Threshold": threshold,
            "Margin": margin,
            "Tong so mau kiem tra": tong_so_mau,
            "Nhan dien dung": so_dung,
            "Nhan dien sai": so_sai,
            "Khong nhan dien": so_khong_nhan_dien,
            "So mau bo qua": so_mau_bo_qua,
            "Ti le dung (%)": round(ti_le_dung, 2),
            "Ti le sai (%)": round(ti_le_sai, 2),
            "Ti le khong nhan dien (%)": round(ti_le_khong_nhan_dien, 2),
            "Nhan xet": nhan_xet
        })
    bang_5_3 = pd.DataFrame(ket_qua_danh_gia)
    print("\nBANG 5.3. DANH GIA ANH HUONG CUA THRESHOLD")
    print(bang_5_3.to_string(index=False))
    return bang_5_3
def main():
    bang_5_1 = tao_bang_5_1()
    bang_5_2 = tao_bang_5_2()
    bang_5_3 = tao_bang_5_3()
if __name__ == "__main__":
    main() 