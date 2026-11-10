import hashlib

import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer

from src.data_processor import (
    read_csv_file,
    get_text_columns,
    get_data_quality,
    prepare_documents
)

from src.bertopic_engine import (
    MODEL_NAME,
    fit_bertopic
)

from src.topic_labeler import generate_topic_labels


st.set_page_config(
    page_title="Chatbot Topic Explorer",
    page_icon="🔎",
    layout="wide"
)


# ==================================
# LOAD EMBEDDING MODEL
# ==================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )


# ==================================
# HEADER
# ==================================

st.title("Chatbot Topic Explorer")

st.caption(
    "Ứng dụng khám phá chủ đề hội thoại "
    "bằng thuật toán BERTopic."
)

st.divider()


# ==================================
# DATA UPLOAD
# ==================================

st.subheader("01. Dữ liệu đầu vào")

uploaded_file = st.file_uploader(
    "Tải lên file dữ liệu hội thoại",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Vui lòng tải lên file CSV.")
    st.stop()


try:
    dataframe = read_csv_file(uploaded_file)

except Exception as error:
    st.error(f"Không thể đọc dữ liệu: {error}")
    st.stop()


st.success(
    f"Đọc thành công: {uploaded_file.name}"
)


# ==================================
# COLUMN SELECTION
# ==================================

st.subheader("02. Cấu hình dữ liệu")

text_columns = get_text_columns(dataframe)

if not text_columns:
    st.error("Không tìm thấy cột văn bản.")
    st.stop()


if "noi_dung_khach_hang" in text_columns:
    default_index = text_columns.index(
        "noi_dung_khach_hang"
    )
else:
    default_index = 0


text_column = st.selectbox(
    "Chọn cột nội dung đưa vào BERTopic",
    options=text_columns,
    index=default_index
)

st.caption(
    "Chọn cột chứa nội dung hội thoại mà bạn muốn khám phá chủ đề."
)


# ==================================
# DATA QUALITY
# ==================================

quality = get_data_quality(
    dataframe,
    text_column
)

documents, valid_dataframe = prepare_documents(
    dataframe,
    text_column
)


st.divider()

st.subheader("03. Tổng quan dữ liệu")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Tổng hội thoại", f"{len(dataframe):,}")
c2.metric("Số cột", len(dataframe.columns))
c3.metric("Văn bản hợp lệ", f"{len(documents):,}")
c4.metric("Văn bản trống", quality["empty_text"])

c5, c6, c7 = st.columns(3)

c5.metric("Ô dữ liệu thiếu", quality["missing_cells"])
c6.metric("Nội dung trùng", quality["duplicate_text"])
c7.metric("Dòng trùng hoàn toàn", quality["duplicate_rows"])


# ==================================
# RESET OLD ANALYSIS WHEN DATA CHANGES
# ==================================

file_hash = hashlib.sha256(
    uploaded_file.getvalue()
).hexdigest()

analysis_signature = f"{file_hash}:{text_column}"

if st.session_state.get("analysis_signature") != analysis_signature:

    for key in [
        "topic_user_labels",
        "topic_auto_labels",
        "all_topics_zip_bytes",
        "vn_keywords_refined",
        "topic_info_before_refinement",
        "topic_model",
        "topic_info",
        "result_dataframe",
        # Thông tin kỹ thuật nội bộ không cần lưu để hiển thị trên giao diện.
    ]:
        st.session_state.pop(key, None)

    for key in list(st.session_state):
        if str(key).startswith("topic_user_label_"):
            st.session_state.pop(key, None)
    st.session_state["analysis_signature"] = analysis_signature


# ==================================
# DATA PREVIEW
# ==================================

st.divider()

st.subheader("04. Xem trước dữ liệu")

st.dataframe(
    dataframe.head(10),
    width="stretch",
    hide_index=True
)


with st.expander("Kiểm tra chi tiết các cột"):

    column_report = pd.DataFrame({
        "Tên cột": dataframe.columns,
        "Kiểu dữ liệu": [
            str(dataframe[column].dtype)
            for column in dataframe.columns
        ],
        "Giá trị thiếu": [
            int(dataframe[column].isna().sum())
            for column in dataframe.columns
        ]
    })

    st.dataframe(
        column_report,
        width="stretch",
        hide_index=True
    )


# ==================================
# RUN BERTOPIC
# ==================================

st.divider()

st.subheader("05. Phân tích chủ đề bằng BERTopic")

if len(documents) < 20:

    st.warning(
        "Cần tối thiểu 20 văn bản để bắt đầu thực nghiệm."
    )

else:

    run_button = st.button(
        "Bắt đầu phân tích chủ đề",
        type="primary",
        width="stretch"
    )

    if run_button:

        st.session_state.pop("topic_user_labels", None)
        st.session_state.pop("topic_auto_labels", None)
        st.session_state.pop("all_topics_zip_bytes", None)
        for key in list(st.session_state):
            if str(key).startswith("topic_user_label_"):
                st.session_state.pop(key, None)
        st.session_state.pop("vn_keywords_refined", None)
        st.session_state.pop("topic_info_before_refinement", None)

        try:

            # Load embedding model
            with st.spinner(
                "Đang tải mô hình ngôn ngữ..."
            ):
                embedding_model = load_embedding_model()


            # Generate embeddings
            with st.spinner(
                "Đang xử lý nội dung hội thoại..."
            ):
                embeddings = embedding_model.encode(
                    documents,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )


            # Run BERTopic
            with st.spinner(
                "Đang khám phá các chủ đề trong hội thoại..."
            ):
                (
                    topic_model,
                    topics,
                    probabilities,
                    topic_info
                ) = fit_bertopic(
                    documents,
                    embeddings,
                    embedding_model
                )


            # Attach topic results to original metadata
            result_dataframe = valid_dataframe.copy()

            result_dataframe["topic_id"] = topics

            representation_map = dict(
                zip(
                    topic_info["Topic"],
                    topic_info["Representation"]
                )
            )

            auto_labels = generate_topic_labels(
                topic_info,
                result_dataframe,
                text_column=text_column
            )

            result_dataframe["topic_name"] = (
                result_dataframe["topic_id"].map(auto_labels)
            )

            # Đồng bộ nhãn gợi ý vào BERTopic nếu phiên bản hỗ trợ.
            if hasattr(topic_model, "set_topic_labels"):
                topic_model.set_topic_labels(auto_labels)

            result_dataframe["topic_keywords"] = (
                result_dataframe["topic_id"].map(
                    lambda topic_id: ", ".join(
                        representation_map.get(topic_id, [])
                    )
                )
            )


            # Save results in session
            st.session_state["topic_model"] = topic_model
            st.session_state["topic_auto_labels"] = auto_labels
            st.session_state["topic_info"] = topic_info
            st.session_state["result_dataframe"] = result_dataframe
            st.success(
                "Phân tích chủ đề đã hoàn tất."
            )

        except Exception as error:

            st.error(
                f"Không thể hoàn thành phân tích: {error}"
            )

            st.exception(error)


# STEP5_VIETNAMESE_REPRESENTATION_BEGIN
if (
    st.session_state.get("analysis_signature") == analysis_signature
    and "topic_model" in st.session_state
    and "result_dataframe" in st.session_state
):
    with st.expander("Cải thiện từ khóa tiếng Việt (tùy chọn)", expanded=False):
        st.write(
            "Cập nhật cách hiển thị từ khóa tiếng Việt để các chủ đề dễ đọc và dễ nhận biết hơn."
        )
        if st.session_state.get("vn_keywords_refined"):
            st.success("Đã cập nhật từ khóa tiếng Việt cho các chủ đề.")
        else:
            if st.button("Cải thiện từ khóa tiếng Việt", key="refine_vietnamese_keywords"):
                from src.vietnamese_keywords import refine_topic_representation, display_keyword

                try:
                    fitted_model = st.session_state["topic_model"]
                    current_df = st.session_state["result_dataframe"]
                    fitted_documents = documents
                    if len(fitted_documents) != len(current_df):
                        raise ValueError("Số văn bản và bảng kết quả không khớp.")
                    old_topic_info = st.session_state["topic_info"].copy(deep=True)
                    with st.spinner("Đang cập nhật từ khóa tiếng Việt..."):
                        refined_info = refine_topic_representation(fitted_model, fitted_documents)

                    repr_map = dict(zip(refined_info["Topic"], refined_info["Representation"]))
                    updated_df = current_df.copy()
                    auto_labels = generate_topic_labels(
                        refined_info,
                        updated_df,
                        text_column=text_column
                    )
                    updated_df["topic_name"] = updated_df["topic_id"].map(auto_labels)
                    if hasattr(fitted_model, "set_topic_labels"):
                        fitted_model.set_topic_labels(auto_labels)
                    updated_df["topic_keywords"] = updated_df["topic_id"].map(
                        lambda topic: ", ".join(
                            display_keyword(word) for word in repr_map.get(topic, [])
                        )
                    )
                    st.session_state["topic_info_before_refinement"] = old_topic_info
                    st.session_state["topic_info"] = refined_info
                    st.session_state["topic_auto_labels"] = auto_labels
                    st.session_state["result_dataframe"] = updated_df
                    st.session_state["vn_keywords_refined"] = True
                    st.session_state.pop("all_topics_zip_bytes", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Chưa thể cải thiện từ khóa: {exc}")
                    st.exception(exc)
        st.caption("Bạn có thể đối chiếu từ khóa trước và sau khi cập nhật.")
# STEP5_VIETNAMESE_REPRESENTATION_END

# STEP9_KEYWORD_COMPARISON_BEGIN
# So sanh bieu dien tu khoa truoc/sau; khong thay doi topic assignment.
if (
    st.session_state.get("analysis_signature") == analysis_signature
    and st.session_state.get("vn_keywords_refined")
    and "topic_info_before_refinement" in st.session_state
    and "topic_info" in st.session_state
):
    before_info = st.session_state["topic_info_before_refinement"].copy(deep=True)
    after_info = st.session_state["topic_info"].copy(deep=True)

    required_comparison_columns = {"Topic", "Count", "Representation"}
    if (
        required_comparison_columns.issubset(before_info.columns)
        and required_comparison_columns.issubset(after_info.columns)
    ):
        before_view = before_info[
            ["Topic", "Count", "Representation"]
        ].rename(
            columns={
                "Count": "Số hội thoại trước",
                "Representation": "Từ khóa trước",
            }
        )
        after_view = after_info[
            ["Topic", "Count", "Representation"]
        ].rename(
            columns={
                "Count": "Số hội thoại sau",
                "Representation": "Từ khóa sau",
            }
        )

        comparison = before_view.merge(
            after_view,
            on="Topic",
            how="outer",
            validate="one_to_one",
        ).sort_values("Topic")

        def _format_topic_keywords(words):
            if isinstance(words, (list, tuple)):
                return ", ".join(
                    str(word).replace("_", " ") for word in words[:10]
                )
            return str(words)

        comparison["Từ khóa trước"] = comparison["Từ khóa trước"].apply(
            _format_topic_keywords
        )
        comparison["Từ khóa sau"] = comparison["Từ khóa sau"].apply(
            _format_topic_keywords
        )

        same_topic_ids = (
            before_info["Topic"].tolist() == after_info["Topic"].tolist()
        )
        same_topic_counts = (
            comparison["Số hội thoại trước"].equals(
                comparison["Số hội thoại sau"]
            )
        )

        with st.expander(
            "So sánh từ khóa trước và sau cải thiện",
            expanded=True,
        ):
            st.dataframe(
                comparison[
                    [
                        "Topic",
                        "Số hội thoại trước",
                        "Số hội thoại sau",
                        "Từ khóa trước",
                        "Từ khóa sau",
                    ]
                ].rename(columns={"Topic": "Topic ID"}),
                width="stretch",
                hide_index=True,
            )

            if same_topic_ids and same_topic_counts:
                st.success(
                    "Đã cập nhật từ khóa. Số hội thoại trong từng chủ đề vẫn được giữ nguyên."
                )
            else:
                st.warning(
                    "Có thay đổi trong kết quả chủ đề. Vui lòng kiểm tra lại trước khi xuất dữ liệu."
                )

            st.caption(
                "Bảng dưới đây giúp bạn đối chiếu từ khóa trước và sau khi cập nhật."
            )
# STEP9_KEYWORD_COMPARISON_END

# ==================================
# DISPLAY RESULTS
# ==================================

if (
    st.session_state.get("analysis_signature") == analysis_signature
    and "topic_info" in st.session_state
):

    topic_info = st.session_state["topic_info"]

    result_dataframe = st.session_state["result_dataframe"]

    topic_model = st.session_state["topic_model"]

    topic_count = int(
        topic_info.loc[
            topic_info["Topic"] != -1,
            "Topic"
        ].nunique()
    )

    outlier_count = int(
        (result_dataframe["topic_id"] == -1).sum()
    )

    outlier_rate = (
        outlier_count / len(result_dataframe) * 100
    )


    st.divider()

    st.subheader("06. Kết quả khám phá chủ đề")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Hội thoại phân tích",
        f"{len(result_dataframe):,}"
    )

    r2.metric(
        "Chủ đề phát hiện",
        topic_count
    )

    r3.metric(
        "Chưa phân nhóm",
        f"{outlier_count:,}"
    )

    r4.metric(
        "Tỷ lệ chưa phân nhóm",
        f"{outlier_rate:.1f}%"
    )


    # Topic keywords table
    st.subheader("Danh sách chủ đề")

    topic_display = topic_info.copy()
    auto_labels = st.session_state.get("topic_auto_labels", {})
    topic_display["Tên gợi nhớ (tự đặt)"] = topic_display["Topic"].map(
        lambda topic_id: st.session_state.get("topic_user_labels", {}).get(int(topic_id), "")
    )
    topic_display["Tên mặc định BERTopic"] = (
        topic_display["Name"].astype(str).str.replace("_", " ", regex=False)
    )
    topic_display["Tên chủ đề tự động"] = topic_display["Topic"].map(
        lambda topic_id: auto_labels.get(
            int(topic_id),
            "Hội thoại chưa xác định chủ đề" if int(topic_id) == -1 else f"Chủ đề {int(topic_id)}"
        )
    )

    topic_display["Representation"] = (
        topic_display["Representation"].apply(
            lambda words: ", ".join(str(word).replace("_", " ") for word in words[:8])
            if isinstance(words, (list, tuple))
            else str(words)
        )
    )

    topic_display = topic_display.rename(
        columns={
            "Topic": "Topic ID",
            "Count": "Số hội thoại",
            "Representation": "Từ khóa đại diện"
        }
    )

    st.dataframe(
        topic_display[
            [
                "Topic ID",
                "Số hội thoại",
                "Tên chủ đề tự động",
                "Tên mặc định BERTopic",
                "Tên gợi nhớ (tự đặt)",
                "Từ khóa đại diện"
            ]
        ],
        width="stretch",
        hide_index=True
    )


    # Topic visualization
    if topic_count > 0:

        st.subheader(
            "Biểu đồ từ khóa của các chủ đề"
        )

        try:

            figure = topic_model.visualize_barchart(
                top_n_topics=min(10, topic_count),
                n_words=8,
                custom_labels=True
            )

            # BERTopic mặc định sử dụng title màu đen trên nền tối.
            figure.update_layout(
                template="plotly_dark",
                title_font_color="#F8FAFC",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(
                figure,
                width="stretch"
            )

        except Exception as error:

            st.warning(
                f"Không thể tạo biểu đồ: {error}"
            )


# ==================================
# SAVE CURRENT DATA
# ==================================

st.session_state["source_dataframe"] = dataframe
st.session_state["valid_dataframe"] = valid_dataframe
st.session_state["documents"] = documents
st.session_state["text_column"] = text_column
st.session_state["source_filename"] = uploaded_file.name





# STEP4_TOPIC_EXPLORER_BEGIN
if (
    st.session_state.get("analysis_signature") == analysis_signature
    and "topic_model" in st.session_state
    and "topic_info" in st.session_state
    and "result_dataframe" in st.session_state
):
    from src.topic_explorer import render_topic_explorer

    render_topic_explorer(
        st.session_state["topic_model"],
        st.session_state["topic_info"],
        st.session_state["result_dataframe"],
    )
# STEP4_TOPIC_EXPLORER_END

# STEP6_TOPIC_EXPLORER_INSTALLED
