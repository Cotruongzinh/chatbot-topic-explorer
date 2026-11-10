# Chatbot Topic Explorer

Ứng dụng Streamlit dùng để khám phá các chủ đề trong dữ liệu hội thoại chatbot tiếng Việt.

## Chức năng

- Tải lên dữ liệu hội thoại dạng CSV.
- Phân tích và khám phá các nhóm chủ đề.
- Xem tên chủ đề, từ khóa và số lượng hội thoại.
- Xem, tìm kiếm và lọc hội thoại trong từng chủ đề.
- Xuất dữ liệu từng chủ đề ra Excel hoặc xuất toàn bộ kết quả thành ZIP.

## Khởi chạy

Mở PowerShell tại thư mục dự án:

```powershell
Set-Location "C:\Projects\chatbot-topic-explorer"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ `http://localhost:8501` nếu trình duyệt không tự mở.

## Cài đặt thư viện

Nếu cần tạo lại môi trường Python:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dữ liệu đầu vào

Ứng dụng nhận file CSV có cột văn bản. Với bộ dữ liệu demo hiện tại, ưu tiên chọn `noi_dung_khach_hang`. Có thể dùng `noi_dung_hoi_thoai` nếu cần phân tích toàn bộ lượt trao đổi.

Dữ liệu demo là dữ liệu tổng hợp phục vụ thực nghiệm, không đại diện cho nhật ký khách hàng thực tế.
