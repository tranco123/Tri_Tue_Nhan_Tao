# Ứng dụng nhận diện khuôn mặt phục vụ điểm danh sinh viên

## 1. Mục tiêu đề tài

Đề tài xây dựng một ứng dụng nhỏ có khả năng nhận diện khuôn mặt sinh viên thông qua camera, sau đó tự động ghi nhận kết quả điểm danh vào file CSV. Ứng dụng sử dụng Python và OpenCV, phù hợp với nội dung Chương 5 về thị giác máy tính.

Tên đề tài khuyến nghị:

**Xây dựng ứng dụng nhận diện khuôn mặt phục vụ điểm danh sinh viên bằng Python và OpenCV**

## 2. Công nghệ sử dụng

- Python
- OpenCV
- NumPy
- Pandas
- YuNet: phát hiện khuôn mặt
- SFace: trích xuất đặc trưng và nhận diện khuôn mặt

## 3. Vì sao chọn YuNet + SFace?

So với cách đơn giản Haar Cascade + LBPH, hướng YuNet + SFace có chất lượng tốt hơn vì:

- YuNet là mô hình phát hiện khuôn mặt dựa trên học sâu, ổn định hơn Haar Cascade.
- SFace trích xuất vector đặc trưng khuôn mặt, giúp so khớp khuôn mặt hiệu quả hơn.
- Không cần tự huấn luyện mô hình CNN nặng.
- Chạy được trên CPU, phù hợp máy cá nhân và demo trên lớp.
- Có thể mở rộng để điểm danh nhiều sinh viên.

## 4. Cấu trúc thư mục

```text
diem_danh_khuon_mat_sface/
│
├── data/
│   └── faces/                  # Ảnh khuôn mặt từng sinh viên
│
├── models/
│   ├── onnx/                   # Chứa 2 mô hình YuNet và SFace
│   └── database/               # Database vector khuôn mặt
│
├── attendance/                 # File điểm danh CSV theo ngày
│
├── src/
│   ├── common.py               # Hàm dùng chung
│   ├── check_camera.py         # Kiểm tra camera
│   ├── capture_faces.py        # Thu thập ảnh khuôn mặt
│   ├── build_database.py       # Tạo database khuôn mặt
│   ├── evaluate_database.py    # Đánh giá nhanh database
│   ├── recognize_attendance.py # Nhận diện và điểm danh bằng webcam
│   └── recognize_image.py      # Nhận diện trên ảnh tĩnh
│
├── download_models.py          # Tải mô hình ONNX
├── requirements.txt
└── README.md
```

## 5. Cài đặt

Mở terminal tại thư mục dự án và chạy:

```bash
python -m pip install -r requirements.txt
```

Sau đó tải mô hình:

```bash
python download_models.py
```

Nếu lệnh `python` không chạy, dùng:

```bash
py -m pip install -r requirements.txt
py download_models.py
```

## 6. Kiểm tra camera

```bash
python src/check_camera.py --camera 0
```

Nếu không mở được camera, thử:

```bash
python src/check_camera.py --camera 1
```

## 7. Thu thập dữ liệu khuôn mặt

Mỗi sinh viên nên lấy khoảng 40 đến 80 ảnh. Ví dụ:

```bash
python src/capture_faces.py --person_name "Tran Duc Co" --num_samples 50
```

Với sinh viên khác:

```bash
python src/capture_faces.py --person_name "Nguyen Van A" --num_samples 50
python src/capture_faces.py --person_name "Le Thi B" --num_samples 50
```

Lưu ý khi thu thập:

- Mặt đủ sáng.
- Không đứng quá xa camera.
- Nhìn thẳng, sau đó xoay nhẹ trái/phải.
- Không để ảnh quá mờ.
- Mỗi người nên có nhiều góc mặt khác nhau.

## 8. Tạo database khuôn mặt

Sau khi thu thập xong ảnh, chạy:

```bash
python src/build_database.py
```

File database sẽ được lưu tại:

```text
models/database/face_database.npz
```

## 9. Đánh giá nhanh database

```bash
python src/evaluate_database.py --threshold 0.50 --margin 0.06
```

Nếu kết quả thấp, nên thu thập thêm ảnh rõ hơn cho từng người.

## 10. Chạy nhận diện và điểm danh

```bash
https://docs.google.com/spreadsheets/d/10MiadMn7ba0y2r7f0kfTVnDZq_fXEvR-WgMM6anzpzA/edit?usp=drivesdk
```

Khi nhận diện thành công, kết quả điểm danh sẽ được lưu trong thư mục:

```text
attendance/
```

Ví dụ file:

```text
attendance/diem_danh_2026-05-13.csv
```

## 11. Nhận diện bằng ảnh tĩnh

Dùng khi không tiện dùng webcam trên lớp:

```bash
python src/recognize_image.py --image anh_test.jpg
```

Chương trình sẽ tạo ảnh kết quả có hậu tố `_ket_qua.jpg`.

## 12. Cách chỉnh ngưỡng để tăng độ chính xác

Trong file `recognize_attendance.py`, tham số quan trọng nhất là `--threshold`.

- Nếu nhận nhầm người: tăng threshold lên 0.55 hoặc 0.60.
- Nếu đúng người nhưng hay hiện Unknown: giảm threshold xuống 0.45.
- Nếu trong nhóm có người khá giống nhau: tăng `--margin` lên 0.08.

Ví dụ tăng độ chặt:

```bash
python src/recognize_attendance.py --threshold 0.60 --margin 0.08
```

Ví dụ dễ nhận hơn:

```bash
python src/recognize_attendance.py --threshold 0.45 --margin 0.05
```

## 13. Quy trình thuật toán

```text
Camera / Ảnh đầu vào
        ↓
Phát hiện khuôn mặt bằng YuNet
        ↓
Căn chỉnh và cắt khuôn mặt
        ↓
Trích xuất vector đặc trưng bằng SFace
        ↓
So sánh với database khuôn mặt
        ↓
Xác định sinh viên
        ↓
Ghi kết quả điểm danh vào file CSV
```

## 14. Ưu điểm

- Dễ demo trực tiếp.
- Không cần huấn luyện mô hình lớn.
- Có khả năng nhận diện nhiều sinh viên.
- Lưu kết quả điểm danh tự động.
- Có thể chạy trên máy cá nhân.

## 15. Hạn chế

- Kết quả phụ thuộc vào ánh sáng, góc mặt và chất lượng camera.
- Nếu hai khuôn mặt quá giống nhau, cần tăng số ảnh mẫu và tăng ngưỡng nhận diện.
- Chưa có giao diện đồ họa hoàn chỉnh.
- Chưa chống gian lận bằng ảnh chụp hoặc video giả.

## 16. Hướng phát triển

- Thêm giao diện bằng Tkinter hoặc Streamlit.
- Xuất điểm danh ra Excel.
- Thêm chức năng quản lý danh sách sinh viên.
- Thêm kiểm tra chống giả mạo bằng nháy mắt hoặc phát hiện chuyển động.
- Kết nối cơ sở dữ liệu SQLite/MySQL.
