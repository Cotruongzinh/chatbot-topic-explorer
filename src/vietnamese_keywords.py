"""Cải thiện biểu diễn topic tiếng Việt mà không phân cụm lại.

PyVi chỉ dùng ở bước CountVectorizer/c-TF-IDF.
Embedding, UMAP, HDBSCAN và Topic ID được giữ nguyên.
"""

import re
import unicodedata

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


# Chỉ loại từ giao tiếp/chức năng phổ biến.
# CỐ Ý GIỮ các từ có ý nghĩa nghiệp vụ hoặc sắc thái ý định:
# không, chưa, hủy, lỗi, bàn, đặt, đổi, mã, booking, ngày, giờ...
VI_STOPWORDS = frozenset({
    "mình", "em", "anh", "chị", "tôi", "bạn",
    "dạ", "ạ", "vâng", "nha", "nhé", "nhỉ", "ơi", "nè",
    "xin", "chào", "cảm_ơn", "làm_ơn", "xin_chào", "vui_lòng",
    "giúp", "giùm", "dùm", "hộ", "giúp_đỡ",
    "muốn", "cho", "thì", "là", "và", "mà", "của", "như",
    "ở", "tại", "đây", "đó", "này", "kia", "cũng", "rồi",
    "à", "á", "thế", "ạ", "dạ_vâng",
})

_EDGE_PUNCTUATION = ".,!?;:()[]{}\"'“”‘’…/\\|"
_IDENTIFIER_PATTERN = re.compile(r"(?:\d+|m\d+|ht\d+)$", re.IGNORECASE)
_HAS_LETTER_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)


def vietnamese_tokens(text):
    """Tách từ tiếng Việt, chuẩn hóa Unicode và bỏ từ đệm."""
    normalized = unicodedata.normalize("NFC", str(text).lower())
    segmented = ViTokenizer.tokenize(normalized)

    words = []
    for raw_word in segmented.split():
        word = unicodedata.normalize(
            "NFC",
            raw_word.strip(_EDGE_PUNCTUATION)
        )

        if not word or word in VI_STOPWORDS:
            continue

        # Loại số thuần và một số mã kỹ thuật thường gây nhiễu từ khóa.
        if _IDENTIFIER_PATTERN.fullmatch(word):
            continue

        # Giữ token có chữ cái; bỏ dấu câu/ký hiệu và token quá ngắn.
        if len(word) < 2 or not _HAS_LETTER_PATTERN.search(word):
            continue

        words.append(word)

    return words


def build_vietnamese_vectorizer():
    """Vectorizer cho văn bản đã được PyVi tách từ."""
    return CountVectorizer(
        tokenizer=vietnamese_tokens,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
    )


def refine_topic_representation(topic_model, documents):
    """Cập nhật từ khóa/c-TF-IDF; không chạy lại embedding hay clustering."""
    documents = [str(document) for document in documents]

    if not documents:
        raise ValueError("Không có văn bản để cải thiện biểu diễn topic.")

    before_topics = tuple(int(topic) for topic in topic_model.topics_)
    if len(before_topics) != len(documents):
        raise ValueError(
            "Số topic hiện tại không khớp số văn bản đầu vào."
        )

    before_sizes = {
        int(topic): int(size)
        for topic, size in topic_model.topic_sizes_.items()
    }

    vectorizer = build_vietnamese_vectorizer()

    # Kiểm tra trước để báo lỗi rõ ràng nếu stopwords loại hết từ.
    try:
        vectorizer.fit(documents)
    except ValueError as exc:
        raise ValueError(
            "Không còn từ khóa sau khi tách từ và lọc stopwords. "
            "Hãy giảm danh sách stopwords hoặc kiểm tra cột văn bản."
        ) from exc

    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model.update_topics(
        documents,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        top_n_words=10,
    )

    after_topics = tuple(int(topic) for topic in topic_model.topics_)
    after_sizes = {
        int(topic): int(size)
        for topic, size in topic_model.topic_sizes_.items()
    }

    if before_topics != after_topics or before_sizes != after_sizes:
        raise RuntimeError(
            "Topic ID hoặc số lượng hội thoại đã thay đổi. "
            "Dừng sử dụng kết quả này và chạy lại phân tích từ đầu."
        )

    return topic_model.get_topic_info()


def display_keyword(word):
    """Hiển thị cụm từ PyVi có dấu gạch dưới thành cụm từ dễ đọc."""
    normalized = unicodedata.normalize("NFC", str(word))
    return normalized.replace("_", " ")