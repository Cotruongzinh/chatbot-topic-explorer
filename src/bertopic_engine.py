from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def fit_bertopic(documents, embeddings, embedding_model):
    """
    Xây dựng và thực thi BERTopic trên danh sách hội thoại.
    """

    # 1. Giảm chiều embeddings
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    # 2. Phân cụm bằng HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # 3. Trích xuất từ khóa và cụm từ
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2
    )

    # 4. Khởi tạo BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=True
    )

    # 5. Huấn luyện và phân nhóm hội thoại
    topics, probabilities = topic_model.fit_transform(
        documents,
        embeddings
    )

    # 6. Lấy thông tin các chủ đề
    topic_info = topic_model.get_topic_info()

    return topic_model, topics, probabilities, topic_info
