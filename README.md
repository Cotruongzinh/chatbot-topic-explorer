# Chatbot Topic Explorer

Ứng dụng Streamlit hỗ trợ khám phá và phân nhóm chủ đề trong dữ liệu hội thoại chatbot tiếng Việt bằng BERTopic.

## 1. Chức năng chính

- Tải lên dữ liệu hội thoại định dạng CSV.
- Lựa chọn cột văn bản cần phân tích.
- Khám phá các chủ đề bằng BERTopic.
- Hiển thị tên chủ đề, từ khóa đại diện và số lượng hội thoại.
- Trực quan hóa phân bố và từ khóa của các chủ đề.
- Xem, tìm kiếm và lọc hội thoại theo chủ đề.
- Xuất kết quả ra Excel hoặc ZIP.
- Tự động lưu và tải lại mô hình BERTopic phù hợp với dữ liệu.

## 2. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình |
| Streamlit | Xây dựng giao diện |
| BERTopic | Khám phá chủ đề |
| Sentence Transformers | Tạo biểu diễn ngữ nghĩa văn bản |
| UMAP | Giảm chiều embedding |
| HDBSCAN | Phân cụm dữ liệu |
| c-TF-IDF | Trích xuất từ khóa đại diện |
| PyVi | Hỗ trợ xử lý tiếng Việt |
| Pandas | Xử lý dữ liệu |
| Plotly | Trực quan hóa |

## 3. Cài đặt và khởi chạy

Yêu cầu Python 3.11.

Tạo môi trường ảo:

```powershell
py -3.11 -m venv .venv
```

Cài đặt thư viện:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Khởi chạy ứng dụng:

```powershell
.\start.bat
```

Hoặc:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Truy cập:

http://localhost:8501

## 4. Quy trình thực nghiệm

1. Chuẩn bị dữ liệu hội thoại dạng CSV.
2. Tải dữ liệu lên ứng dụng.
3. Chọn cột văn bản cần phân tích, ví dụ `noi_dung_khach_hang`.
4. Nhấn nút **Phân tích chủ đề**.
5. Xem kết quả phân nhóm, từ khóa và biểu đồ trực quan.
6. Chọn từng chủ đề để khám phá các hội thoại tương ứng.
7. Xuất kết quả phục vụ phân tích tiếp theo.

## 5. Cơ chế lưu và tải lại mô hình

Ứng dụng tích hợp cơ chế quản lý mô hình BERTopic ngay trong quy trình phân tích.

- Khi nhấn **Phân tích chủ đề**, ứng dụng kiểm tra mô hình đã lưu có phù hợp với dữ liệu và cột văn bản hiện tại hay không.
- Nếu tìm thấy mô hình phù hợp, ứng dụng tải lại mô hình đã lưu, không cần thực hiện fitting từ đầu.
- Nếu chưa có mô hình phù hợp, ứng dụng thực hiện huấn luyện BERTopic trên dữ liệu hiện tại và tự động lưu mô hình.
- Mô hình được lưu dưới dạng pickle trong thư mục `models/`.
- Thông tin mô hình được quản lý thông qua file metadata.

Embedding model sử dụng:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Ở lần chạy đầu tiên trên máy mới, Sentence Transformers có thể cần kết nối Internet để tải embedding model.

### Lưu ý khi sử dụng

- Thư mục `models/` là artifact được tạo tại máy local và không đưa lên Git.
- Khi clone project trên máy mới, cần chạy phân tích lần đầu để tạo mô hình local.
- Chỉ tải các file pickle được tạo từ nguồn đáng tin cậy.

## 6. Cấu trúc dự án

```text
chatbot-topic-explorer/
├── .streamlit/
│   └── config.toml
├── data/
│   └── sample_conversations.csv
├── src/
│   ├── __init__.py
│   ├── bertopic_engine.py
│   ├── data_processor.py
│   ├── model_storage.py
│   ├── topic_explorer.py
│   ├── topic_labeler.py
│   └── vietnamese_keywords.py
├── .gitattributes
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── start.bat
```

Thư mục `models/` được tạo tự động khi ứng dụng lưu mô hình. Đây là thư mục local và không được đưa lên Git.

## 7. Dữ liệu thực nghiệm

Project cung cấp bộ dữ liệu hội thoại tổng hợp để phục vụ thử nghiệm khám phá chủ đề.

Dữ liệu demo không đại diện cho nhật ký cuộc gọi hoặc hội thoại thực tế của khách hàng.