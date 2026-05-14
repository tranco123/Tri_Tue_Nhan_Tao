# Gợi ý viết báo cáo đề tài

## Tên đề tài

**Xây dựng ứng dụng nhận diện khuôn mặt phục vụ điểm danh sinh viên bằng Python và OpenCV**

## 1. Mở đầu

Trong bối cảnh chuyển đổi số trong giáo dục, việc tự động hóa các hoạt động quản lý lớp học ngày càng được quan tâm. Điểm danh sinh viên là một công việc diễn ra thường xuyên, tuy đơn giản nhưng tốn thời gian nếu thực hiện thủ công. Bên cạnh đó, điểm danh thủ công còn có thể xảy ra sai sót hoặc tình trạng điểm danh hộ. Vì vậy, nhóm lựa chọn xây dựng ứng dụng nhận diện khuôn mặt phục vụ điểm danh sinh viên nhằm minh họa một ứng dụng thực tế của thị giác máy tính.

Đề tài sử dụng ngôn ngữ Python kết hợp thư viện OpenCV để phát hiện và nhận diện khuôn mặt thông qua camera. Khi hệ thống nhận diện được sinh viên, thông tin điểm danh sẽ được lưu tự động vào file CSV gồm họ tên, ngày, giờ và điểm tin cậy.

## 2. Mục tiêu đề tài

### 2.1. Mục tiêu tổng quát

Xây dựng một ứng dụng nhỏ có khả năng phát hiện khuôn mặt, nhận diện sinh viên và tự động ghi nhận kết quả điểm danh, qua đó vận dụng kiến thức về thị giác máy tính vào một bài toán thực tế.

### 2.2. Mục tiêu cụ thể

- Tìm hiểu khái niệm thị giác máy tính và bài toán nhận diện khuôn mặt.
- Thu thập dữ liệu khuôn mặt của các thành viên trong nhóm.
- Tiền xử lý và trích xuất đặc trưng khuôn mặt.
- Xây dựng database khuôn mặt cho từng sinh viên.
- Nhận diện khuôn mặt qua webcam theo thời gian thực.
- Ghi kết quả điểm danh vào file CSV.
- Đánh giá ưu điểm, hạn chế và hướng phát triển của hệ thống.

## 3. Cơ sở lý thuyết

### 3.1. Thị giác máy tính

Thị giác máy tính là lĩnh vực thuộc trí tuệ nhân tạo và khoa học máy tính, nghiên cứu các phương pháp giúp máy tính có khả năng tiếp nhận, xử lý và hiểu thông tin từ hình ảnh hoặc video. Trong bài toán điểm danh, thị giác máy tính được sử dụng để phát hiện vị trí khuôn mặt trong khung hình, sau đó nhận diện khuôn mặt thuộc về sinh viên nào.

### 3.2. Phát hiện khuôn mặt

Phát hiện khuôn mặt là bước xác định trong ảnh có khuôn mặt hay không và khuôn mặt nằm ở vị trí nào. Kết quả thường là một khung bao quanh khuôn mặt. Trong đề tài, nhóm sử dụng mô hình YuNet để phát hiện khuôn mặt vì mô hình này nhẹ, tốc độ nhanh và phù hợp với ứng dụng thời gian thực.

### 3.3. Nhận diện khuôn mặt

Nhận diện khuôn mặt là quá trình xác định danh tính của người xuất hiện trong ảnh hoặc video. Sau khi khuôn mặt được phát hiện, hệ thống sẽ căn chỉnh khuôn mặt và trích xuất vector đặc trưng bằng mô hình SFace. Vector đặc trưng này được so sánh với database đã lưu để tìm ra người có khuôn mặt giống nhất.

### 3.4. Độ tương đồng cosine

Độ tương đồng cosine được sử dụng để đo mức độ giống nhau giữa hai vector đặc trưng khuôn mặt. Giá trị càng cao thì hai khuôn mặt càng giống nhau. Nếu điểm tương đồng vượt qua ngưỡng nhận diện, hệ thống sẽ xác định đó là sinh viên tương ứng; ngược lại, hệ thống hiển thị là Unknown.

## 4. Phân tích và thiết kế hệ thống

### 4.1. Đầu vào và đầu ra

Đầu vào của hệ thống là hình ảnh hoặc video từ webcam. Đầu ra là tên sinh viên được nhận diện trên màn hình và file điểm danh CSV chứa thông tin họ tên, ngày, giờ và điểm tin cậy.

### 4.2. Quy trình hoạt động

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

### 4.3. Các chức năng chính

- Kiểm tra camera.
- Thu thập ảnh khuôn mặt sinh viên.
- Tạo database khuôn mặt.
- Đánh giá nhanh database.
- Nhận diện và điểm danh bằng webcam.
- Nhận diện khuôn mặt trên ảnh tĩnh.

## 5. Cài đặt và thực nghiệm

### 5.1. Môi trường thực nghiệm

- Ngôn ngữ: Python
- Thư viện: OpenCV, NumPy, Pandas
- Thiết bị: Máy tính cá nhân có webcam
- Mô hình: YuNet và SFace định dạng ONNX

### 5.2. Thu thập dữ liệu

Mỗi sinh viên được thu thập khoảng 40 đến 80 ảnh khuôn mặt. Trong quá trình thu thập, sinh viên được yêu cầu nhìn thẳng, xoay nhẹ mặt sang trái/phải, thay đổi biểu cảm nhẹ và đảm bảo điều kiện ánh sáng phù hợp. Việc thu thập đa dạng góc mặt giúp hệ thống nhận diện ổn định hơn.

### 5.3. Tạo database khuôn mặt

Sau khi thu thập dữ liệu, hệ thống đọc từng ảnh khuôn mặt, trích xuất vector đặc trưng và lưu vào database. Mỗi sinh viên có nhiều vector mẫu; hệ thống lấy vector trung bình làm vector đại diện để tăng tính ổn định khi nhận diện.

### 5.4. Nhận diện và điểm danh

Khi chạy webcam, hệ thống phát hiện khuôn mặt trong từng khung hình, trích xuất vector đặc trưng và so sánh với database. Nếu điểm tương đồng đạt ngưỡng, tên sinh viên sẽ được hiển thị trên màn hình và kết quả được ghi vào file CSV. Hệ thống tránh ghi trùng bằng cách kiểm tra sinh viên đã được điểm danh trong ngày hay chưa.

## 6. Kết quả đạt được

Ứng dụng đã thực hiện được các chức năng cơ bản của một hệ thống điểm danh bằng khuôn mặt. Hệ thống có thể mở camera, phát hiện khuôn mặt, nhận diện sinh viên đã có trong database và tự động lưu kết quả điểm danh. Khi điều kiện ánh sáng tốt, khuôn mặt nhìn rõ và dữ liệu mẫu đủ nhiều, hệ thống cho kết quả nhận diện ổn định.

## 7. Ưu điểm

- Giao diện demo trực quan thông qua webcam.
- Có thể nhận diện nhiều sinh viên.
- Tự động lưu kết quả điểm danh.
- Không cần huấn luyện mô hình học sâu phức tạp.
- Phù hợp với phạm vi bài thực hành môn học.

## 8. Hạn chế

- Kết quả phụ thuộc vào ánh sáng, góc mặt và chất lượng camera.
- Nếu dữ liệu mẫu ít hoặc ảnh bị mờ, độ chính xác sẽ giảm.
- Hệ thống chưa có giao diện quản lý hoàn chỉnh.
- Chưa có chức năng chống giả mạo bằng ảnh chụp hoặc video.

## 9. Hướng phát triển

Trong tương lai, hệ thống có thể được phát triển thêm giao diện người dùng, kết nối cơ sở dữ liệu, xuất file Excel, quản lý danh sách lớp học và bổ sung cơ chế chống giả mạo như yêu cầu người dùng nháy mắt, quay đầu hoặc kiểm tra chuyển động thực tế.

## 10. Kết luận

Đề tài đã vận dụng kiến thức thị giác máy tính vào bài toán điểm danh sinh viên. Thông qua việc sử dụng OpenCV, YuNet và SFace, nhóm đã xây dựng được một ứng dụng có khả năng phát hiện, nhận diện khuôn mặt và ghi nhận kết quả điểm danh tự động. Mặc dù còn một số hạn chế, hệ thống vẫn đáp ứng được mục tiêu đặt ra và có khả năng mở rộng trong các ứng dụng quản lý lớp học thực tế.
