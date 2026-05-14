# Kịch bản demo trên lớp

## 1. Chuẩn bị trước khi demo

Trước buổi thuyết trình, nhóm nên chuẩn bị:

- Máy tính đã cài Python.
- Webcam hoạt động tốt.
- Đã tải mô hình ONNX.
- Đã thu thập ảnh khuôn mặt của 2 đến 5 thành viên.
- Đã chạy `build_database.py` trước.
- Đã kiểm tra file điểm danh trong thư mục `attendance`.

## 2. Thứ tự demo khuyến nghị

### Bước 1: Giới thiệu mục tiêu

Nhóm trình bày ngắn gọn:

> Ứng dụng của nhóm có nhiệm vụ nhận diện khuôn mặt sinh viên qua webcam và tự động ghi kết quả điểm danh vào file CSV.

### Bước 2: Mở cấu trúc thư mục

Cho thầy thấy các thư mục chính:

- `data/faces`: lưu ảnh khuôn mặt.
- `models/onnx`: lưu mô hình YuNet và SFace.
- `models/database`: lưu database khuôn mặt.
- `attendance`: lưu kết quả điểm danh.
- `src`: chứa mã nguồn.

### Bước 3: Demo dữ liệu khuôn mặt

Mở thư mục `data/faces` để cho thấy mỗi người có một thư mục riêng, bên trong là các ảnh khuôn mặt đã thu thập.

### Bước 4: Chạy nhận diện webcam

Chạy lệnh:

```bash
python src/recognize_attendance.py --threshold 0.50 --margin 0.06
```

Sau đó từng thành viên đưa mặt vào camera để hệ thống nhận diện.

### Bước 5: Mở file điểm danh

Sau khi nhận diện thành công, mở thư mục `attendance` và mở file CSV theo ngày để cho thấy hệ thống đã lưu:

- Họ tên
- Ngày
- Thời gian
- Điểm tin cậy

## 3. Câu nói khi thuyết trình

> Đầu tiên, hệ thống sử dụng camera để lấy khung hình đầu vào. Sau đó, mô hình YuNet được dùng để phát hiện vị trí khuôn mặt. Khuôn mặt được căn chỉnh và đưa vào mô hình SFace để trích xuất vector đặc trưng. Vector này được so sánh với database khuôn mặt đã lưu trước đó. Nếu điểm tương đồng đạt ngưỡng, hệ thống xác định danh tính sinh viên và ghi kết quả điểm danh vào file CSV.

## 4. Câu hỏi thầy có thể hỏi và cách trả lời

### Câu 1: Vì sao nhóm không dùng Haar Cascade?

Trả lời:

> Haar Cascade dễ triển khai nhưng độ ổn định chưa cao trong các điều kiện ánh sáng và góc mặt khác nhau. Nhóm chọn YuNet vì đây là mô hình phát hiện khuôn mặt dựa trên học sâu, nhẹ và phù hợp với xử lý thời gian thực.

### Câu 2: Hệ thống nhận diện bằng cách nào?

Trả lời:

> Hệ thống dùng SFace để biến mỗi khuôn mặt thành một vector đặc trưng. Khi có khuôn mặt mới, hệ thống cũng trích xuất vector và so sánh với các vector đã lưu bằng độ tương đồng cosine.

### Câu 3: Nếu có người lạ thì sao?

Trả lời:

> Nếu điểm tương đồng thấp hơn ngưỡng nhận diện, hệ thống sẽ hiển thị Unknown và không ghi điểm danh.

### Câu 4: Làm sao giảm nhận nhầm?

Trả lời:

> Có thể tăng số lượng ảnh mẫu cho mỗi sinh viên, thu thập ảnh trong điều kiện ánh sáng tốt hơn, tăng ngưỡng nhận diện và dùng thêm ngưỡng chênh lệch giữa người giống nhất với người giống thứ hai.

### Câu 5: Hạn chế lớn nhất là gì?

Trả lời:

> Hạn chế lớn nhất là hệ thống chưa có chống giả mạo, ví dụ một người có thể đưa ảnh khuôn mặt trước camera. Hướng phát triển là thêm kiểm tra nháy mắt, quay đầu hoặc phát hiện chuyển động thực.
