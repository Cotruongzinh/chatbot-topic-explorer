"""Tạo tên topic tiếng Việt tự động từ từ khóa và hội thoại mẫu.

Đây là bộ gợi ý nhẹ, chạy offline, không gọi API và không thay đổi kết quả
phân cụm BERTopic. Người dùng vẫn có thể chỉnh sửa tên gợi nhớ trên giao diện.
"""

import re
import unicodedata


def _normalize(value):
    """Chuẩn hóa tiếng Việt để so khớp từ khóa không phụ thuộc dấu."""
    text = unicodedata.normalize("NFD", str(value or "").lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Các luật được sắp xếp từ chủ đề đặc thù đến chủ đề tổng quát.
# Có thể bổ sung luật mới khi thay đổi lĩnh vực chatbot.
_TOPIC_RULES = [
    (
        "Thông tin nhà hàng tại Estella Place",
        ("estella place", "estella"),
        3,
    ),
    (
        "Thông tin chi nhánh AEON Mall Tân Phú",
        ("aeon mall", "tan phu", "mall tan phu"),
        3,
    ),
    (
        "Tư vấn món ăn và dị ứng thực phẩm",
        ("di ung", "nguoi di ung", "an toan thuc pham"),
        3,
    ),
    (
        "Yêu cầu ghế ngồi cho trẻ nhỏ",
        ("ghe be", "ghe cho be", "tre nho", "tre em", "co be"),
        3,
    ),
    (
        "Tư vấn món pizza và thành phần",
        ("pizza", "mon pizza", "thanh phan", "sot kem"),
        3,
    ),
    (
        "Quy định mang đồ ăn và đặt món",
        ("mang banh", "mang do an", "quy dinh mang", "dat mon", "phi ghi"),
        3,
    ),
    (
        "Tìm kiếm địa chỉ và chi nhánh nhà hàng",
        ("chi nhanh", "cua hang", "dia chi", "khu vuc"),
        5,
    ),
    (
        "Tra cứu và quản lý thông tin booking",
        ("booking", "ma booking", "ma mo2", "trang thai", "kiem tra", "huy booking", "thay doi booking"),
        5,
    ),
    (
        "Tìm bàn và đặt bàn theo ngày giờ",
        ("dat ban", "ban trong", "ngay", "gio", "so khach", "nhom khach"),
        5,
    ),
]

_FALLBACK_STOPWORDS = {
    "minh", "em", "anh", "chi", "toi", "ban", "co", "khong", "cho",
    "nguoi", "la", "va", "cac", "nhung", "duoc", "can", "muon",
    "mot", "hai", "nay", "do", "roi", "nha", "nhe", "a", "da",
    "topic", "moi", "se", "thi", "ve", "tai", "theo",
}


def _as_terms(value):
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _score_rule(keyword_text, document_texts, patterns):
    """Chấm điểm luật, ưu tiên từ khóa topic hơn hội thoại mẫu.

    Từ khóa c-TF-IDF là tín hiệu chính để đặt tên. Hội thoại mẫu chỉ dùng
    để xác nhận thêm; tổng điểm từ hội thoại mẫu bị giới hạn để một vài câu
    có nhắc thoáng qua đến chủ đề khác không làm gán nhầm tên topic.
    """
    keyword_score = 0
    document_score = 0
    normalized_docs = [_normalize(doc) for doc in document_texts]

    for pattern in patterns:
        pattern = _normalize(pattern)
        if not pattern:
            continue

        if pattern in keyword_text:
            keyword_score += 3

        if any(pattern in doc for doc in normalized_docs):
            document_score += 1

    # Tối đa 2 điểm hỗ trợ từ văn bản mẫu. Vì mọi luật cần ít nhất 3 điểm,
    # văn bản mẫu không thể tự mình quyết định tên topic.
    return keyword_score + min(document_score, 2)


def _fallback_label(terms):
    cleaned = []
    seen = set()

    for term in terms:
        normalized = _normalize(term)
        if not normalized or normalized in _FALLBACK_STOPWORDS:
            continue
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(str(term).replace("_", " ").strip())

    if not cleaned:
        return "Chủ đề hội thoại khác"

    return "Chủ đề về " + ", ".join(cleaned[:3])


def generate_topic_labels(topic_info, result_dataframe, text_column=None, max_docs=5):
    """Trả về dict {topic_id: tên tiếng Việt} từ từ khóa và hội thoại mẫu.

    Topic -1 luôn được đặt tên là "Hội thoại chưa xác định chủ đề".
    Hàm chỉ đọc kết quả hiện có; không chạy lại embedding hay clustering.
    """
    labels = {}

    if topic_info is None or len(topic_info) == 0:
        return labels

    dataframe_columns = getattr(result_dataframe, "columns", [])
    if text_column not in dataframe_columns:
        candidates = ("noi_dung_khach_hang", "noi_dung_hoi_thoai")
        text_column = next(
            (column for column in candidates if column in dataframe_columns),
            None,
        )

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])

        if topic_id == -1:
            labels[topic_id] = "Hội thoại chưa xác định chủ đề"
            continue

        terms = _as_terms(row.get("Representation", []))
        keyword_text = _normalize(" ".join(terms))

        docs = []
        if (
            result_dataframe is not None
            and "topic_id" in result_dataframe.columns
            and text_column is not None
        ):
            topic_rows = result_dataframe.loc[
                result_dataframe["topic_id"].astype(int) == topic_id,
                text_column,
            ]
            docs = (
                topic_rows.dropna()
                .astype(str)
                .map(str.strip)
            )
            docs = [doc for doc in docs.tolist() if doc][:max_docs]

        best_label = None
        best_score = 0

        for label, patterns, minimum_score in _TOPIC_RULES:
            score = _score_rule(keyword_text, docs, patterns)
            if score >= minimum_score and score > best_score:
                best_label = label
                best_score = score

        if best_label is not None:
            labels[topic_id] = best_label
        else:
            labels[topic_id] = _fallback_label(terms)

    return labels
