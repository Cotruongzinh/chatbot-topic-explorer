"""Giao diện khám phá topic và xuất báo cáo hội thoại.

Phần này dùng các topic đã được BERTopic tạo; không huấn luyện hay phân cụm lại.
"""
from io import BytesIO
import hashlib
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import plotly.express as px
import streamlit as st
from openpyxl.styles import Alignment, Font, PatternFill

EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _excel_safe(value):
    """Chống công thức Excel đến từ CSV hoặc nhãn do người dùng nhập."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _style_excel(workbook):
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="20634F")
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.row_dimensions[1].height = 30
        for cells in sheet.columns:
            header = cells[0]
            name = str(header.value or "")
            width = 50 if any(word in name.lower() for word in ("noi_dung", "giá trị", "từ khóa")) else 26
            sheet.column_dimensions[header.column_letter].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)


def _safe_frame(frame):
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(_excel_safe)
    return result


def _keywords(topic_model, topic_id):
    if topic_id == -1:
        return "Chưa phân nhóm"
    terms = topic_model.get_topic(int(topic_id)) or []
    return ", ".join(str(word).replace("_", " ") for word, _ in terms[:10])


def _topic_name(topic_id, labels):
    """Tên gợi nhớ do người xem tự nhập."""
    if int(topic_id) == -1:
        return "Chưa phân nhóm"
    return labels.get(int(topic_id), "")


def _automatic_topic_name(topic_id, automatic_labels):
    """Tên chủ đề được gợi ý tự động từ từ khóa và hội thoại mẫu."""
    if int(topic_id) == -1:
        return "Hội thoại chưa xác định chủ đề"
    return automatic_labels.get(int(topic_id), f"Chủ đề {int(topic_id)}")


def _display_topic_name(topic_id, user_labels, automatic_labels):
    """Ưu tiên tên người xem tự đặt; nếu chưa có thì dùng tên tự động."""
    if int(topic_id) == -1:
        return "Hội thoại chưa xác định chủ đề"
    return (
        user_labels.get(int(topic_id))
        or _automatic_topic_name(topic_id, automatic_labels)
    )


def create_topic_excel(topic_id, selected, topic_info, topic_model, labels=None):
    """Tạo Excel cho TOÀN BỘ hội thoại của một topic, không bị ảnh hưởng bộ lọc UI."""
    labels = labels or {}
    info = topic_info.loc[topic_info["Topic"].astype(int) == int(topic_id)]
    if info.empty:
        raise ValueError(f"Không tồn tại Topic {topic_id} trong bảng kết quả.")
    row = info.iloc[0]
    bertopic_name = str(row["Name"]).replace("_", " ")
    if "topic_name" in selected.columns and selected["topic_name"].notna().any():
        automatic_name = str(selected["topic_name"].dropna().iloc[0])
    else:
        automatic_name = _automatic_topic_name(topic_id, {})
    user_name = _topic_name(topic_id, labels)

    overview = pd.DataFrame([
        {"Chỉ số": "Topic ID", "Giá trị": int(topic_id)},
        {"Chỉ số": "Tên chủ đề tự động", "Giá trị": automatic_name},
        {"Chỉ số": "Tên mặc định BERTopic", "Giá trị": bertopic_name},
        {"Chỉ số": "Tên gợi nhớ do người xem đặt", "Giá trị": user_name},
        {"Chỉ số": "Số hội thoại", "Giá trị": len(selected)},
        {"Chỉ số": "Từ khóa đại diện", "Giá trị": _keywords(topic_model, topic_id)},
        {"Chỉ số": "Nguồn phân cụm", "Giá trị": "BERTopic / HDBSCAN"},
        {"Chỉ số": "Nhãn tham chiếu", "Giá trị": "Chỉ để đối chiếu; không dùng khi phân cụm"},
    ])
    overview["Giá trị"] = overview["Giá trị"].map(_excel_safe)

    preferred = [
        "ma_hoi_thoai", "noi_dung_khach_hang", "noi_dung_hoi_thoai",
        "kenh", "chu_de_tham_chieu", "kich_ban_nguon",
        "topic_id", "topic_name", "topic_keywords"
    ]
    columns = [col for col in preferred if col in selected.columns]
    others = [col for col in selected.columns if col not in columns]
    rows = selected[columns + others].copy()
    rows["ten_topic_do_nguoi_xem_dat"] = user_name
    rows = _safe_frame(rows)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="ThongTinTopic", index=False)
        rows.to_excel(writer, sheet_name="HoiThoai", index=False)
        _style_excel(writer.book)
    return out.getvalue()


def create_all_topics_zip(topic_info, result_dataframe, topic_model, labels=None):
    """Tạo ZIP gồm 1 Excel tổng hợp và 1 Excel riêng cho từng topic (cả -1)."""
    labels = labels or {}
    output = BytesIO()
    overview = topic_info[["Topic", "Count", "Name"]].copy()
    overview["Name"] = overview["Name"].astype(str).str.replace("_", " ", regex=False)

    automatic_labels = {}
    if "topic_name" in result_dataframe.columns:
        label_rows = result_dataframe[["topic_id", "topic_name"]].dropna().drop_duplicates("topic_id")
        automatic_labels = {
            int(row["topic_id"]): str(row["topic_name"])
            for _, row in label_rows.iterrows()
        }

    overview["Tên chủ đề tự động"] = overview["Topic"].map(
        lambda topic_id: _automatic_topic_name(int(topic_id), automatic_labels)
    )
    overview["Tên gợi nhớ (tự đặt)"] = overview["Topic"].map(
        lambda topic_id: _topic_name(topic_id, labels)
    )
    overview["Từ khóa đại diện"] = overview["Topic"].map(
        lambda topic_id: _keywords(topic_model, int(topic_id))
    )
    overview = overview.rename(
        columns={
            "Topic": "Topic ID",
            "Count": "Số hội thoại",
            "Name": "Tên mặc định BERTopic",
        }
    )
    overview = _safe_frame(overview)

    summary = BytesIO()
    with pd.ExcelWriter(summary, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="TongHop", index=False)
        _style_excel(writer.book)

    ids = topic_info["Topic"].astype(int).tolist()
    assigned = result_dataframe["topic_id"].astype(int)
    total = 0
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("tong_hop_topics.xlsx", summary.getvalue())
        for topic_id in ids:
            selected = result_dataframe.loc[assigned == topic_id].copy()
            expected = int(topic_info.loc[topic_info["Topic"].astype(int) == topic_id, "Count"].iloc[0])
            if len(selected) != expected:
                raise ValueError(f"Số hội thoại Topic {topic_id} không khớp: {len(selected)} / {expected}.")
            archive.writestr(
                f"topic_{topic_id}_hoi_thoai.xlsx",
                create_topic_excel(topic_id, selected, topic_info, topic_model, labels)
            )
            total += len(selected)
    if total != len(result_dataframe):
        raise ValueError("Có hội thoại không nằm trong bất kỳ file topic nào.")
    return output.getvalue()


def _save_user_label(topic_id, input_key):
    labels = dict(st.session_state.get("topic_user_labels", {}))
    value = str(st.session_state.get(input_key, "")).strip()
    if value:
        labels[int(topic_id)] = value[:100]
    else:
        labels.pop(int(topic_id), None)
    st.session_state["topic_user_labels"] = labels
    st.session_state.pop("all_topics_zip_bytes", None)


def render_topic_explorer(topic_model, topic_info, result_dataframe):
    st.divider()
    st.subheader("07. Khám phá chi tiết từng chủ đề")
    labels = st.session_state.get("topic_user_labels", {})

    automatic_labels = {}
    if "topic_name" in result_dataframe.columns:
        label_rows = result_dataframe[["topic_id", "topic_name"]].dropna().drop_duplicates("topic_id")
        automatic_labels = {
            int(row["topic_id"]): str(row["topic_name"])
            for _, row in label_rows.iterrows()
        }

    plot = topic_info[["Topic", "Count"]].copy().sort_values("Topic")
    plot["Nhãn"] = plot["Topic"].map(
        lambda topic_id: (
            f"Chưa xác định (-1) — {_display_topic_name(topic_id, labels, automatic_labels)}"
            if int(topic_id) == -1
            else f"Topic {int(topic_id)} — {_display_topic_name(topic_id, labels, automatic_labels)}"
        )
    )
    plot["Tên gợi nhớ"] = plot["Topic"].map(lambda t: _topic_name(t, labels))
    chart = px.bar(
        plot, x="Nhãn", y="Count", text="Count", hover_data=["Tên gợi nhớ"],
        labels={"Nhãn": "Chủ đề", "Count": "Số hội thoại"},
        title="Phân bố hội thoại theo chủ đề"
    )
    chart.update_layout(
        template="plotly_dark", showlegend=False, height=370,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font_color="#F8FAFC", margin=dict(l=12, r=12, t=50, b=10)
    )
    chart.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(chart, width="stretch")

    choices = sorted(int(t) for t in topic_info["Topic"] if int(t) >= 0)
    if -1 in topic_info["Topic"].values:
        choices.append(-1)
    counts = dict(zip(topic_info["Topic"].astype(int), topic_info["Count"].astype(int)))
    selected_topic = st.selectbox(
        "Chọn chủ đề để xem nội dung", choices,
        format_func=lambda t: (
            f"Chưa xác định chủ đề (-1) — {counts.get(-1, 0)} hội thoại" if t == -1 else
            f"Topic {t} — {_display_topic_name(t, labels, automatic_labels)} — {counts[t]} hội thoại"
        ),
        key="topic_explorer_selected_topic"
    )
    selected = result_dataframe.loc[result_dataframe["topic_id"].astype(int) == selected_topic].copy()
    if selected.empty:
        st.warning("Chủ đề được chọn không có hội thoại.")
        return

    st.metric("Số hội thoại trong chủ đề", len(selected))
    if selected_topic == -1:
        st.caption("Các hội thoại này chưa được xếp vào một chủ đề cụ thể.")
    else:
        st.markdown("**Tên chủ đề tự động**")
        st.write(_automatic_topic_name(selected_topic, automatic_labels))

        st.markdown("**Từ khóa đại diện**")
        st.write(_keywords(topic_model, selected_topic) or "Không có từ khóa")

        st.caption(
            "Bạn có thể nhập tên gợi nhớ riêng để dễ nhận biết chủ đề."
        )
        signature = str(st.session_state.get("analysis_signature", ""))[:12]
        name_key = f"topic_user_label_{signature}_{selected_topic}"
        if name_key not in st.session_state:
            st.session_state[name_key] = labels.get(selected_topic, "")
        st.text_input(
            "Tên gợi nhớ cho chủ đề (không bắt buộc)",
            key=name_key,
            on_change=_save_user_label,
            args=(selected_topic, name_key),
            placeholder="Ví dụ: Tìm bàn và đặt bàn theo ngày giờ",
            max_chars=100,
        )

    st.markdown("**Tìm kiếm và lọc hội thoại**")
    col_search, col_channel = st.columns([3, 1])
    with col_search:
        query = st.text_input(
            "Tìm theo mã hoặc nội dung", value="", placeholder="Nhập từ khóa, mã hội thoại...",
            key=f"topic_search_{selected_topic}"
        ).strip()
    with col_channel:
        if "kenh" in selected.columns:
            channels = ["Tất cả"] + sorted(selected["kenh"].dropna().astype(str).unique().tolist())
            channel = st.selectbox("Kênh", channels, key=f"topic_channel_{selected_topic}")
        else:
            channel = "Tất cả"

    filtered = selected.copy()
    if channel != "Tất cả" and "kenh" in filtered.columns:
        filtered = filtered.loc[filtered["kenh"].astype(str) == channel]
    if query:
        search_cols = [c for c in ("ma_hoi_thoai", "noi_dung_khach_hang", "noi_dung_hoi_thoai") if c in filtered.columns]
        if search_cols:
            mask = pd.Series(False, index=filtered.index)
            for col in search_cols:
                mask = mask | filtered[col].fillna("").astype(str).str.contains(query, case=False, regex=False)
            filtered = filtered.loc[mask]
    st.caption(f"Đang hiển thị {len(filtered):,}/{len(selected):,} hội thoại trong Topic {selected_topic}.")

    columns = [c for c in ("ma_hoi_thoai", "noi_dung_khach_hang", "kenh", "chu_de_tham_chieu") if c in filtered]
    if not columns:
        columns = filtered.columns[:3].tolist()
    st.dataframe(filtered[columns], width="stretch", hide_index=True, height=310)

    if not filtered.empty:
        ids = filtered.index.tolist()
        id_column = "ma_hoi_thoai" if "ma_hoi_thoai" in filtered else None
        filter_token = hashlib.sha1(
            f"{selected_topic}|{channel}|{query}".encode("utf-8")
        ).hexdigest()[:12]
        row_idx = st.selectbox(
            "Mở một cuộc hội thoại để đọc đầy đủ", ids,
            format_func=lambda i: str(filtered.at[i, id_column]) if id_column else f"Hội thoại {i + 1}",
            key=f"topic_record_selection_{selected_topic}_{filter_token}"
        )
        record = filtered.loc[row_idx]
        if "noi_dung_khach_hang" in filtered:
            st.text_area("Nội dung khách hàng", str(record["noi_dung_khach_hang"]), height=155, disabled=True,
                         key=f"topic_customer_{selected_topic}_{row_idx}")
        if "noi_dung_hoi_thoai" in filtered:
            st.text_area("Toàn bộ hội thoại (khách hàng và chatbot)", str(record["noi_dung_hoi_thoai"]), height=240,
                         disabled=True, key=f"topic_full_{selected_topic}_{row_idx}")
        if "chu_de_tham_chieu" in filtered:
            st.caption("Nhãn tham chiếu từ dữ liệu giả lập: " + str(record["chu_de_tham_chieu"]))
    else:
        st.info("Không có hội thoại khớp bộ lọc. Hãy thử từ khóa khác hoặc chọn 'Tất cả'.")

    st.markdown("**Xuất Excel và toàn bộ kết quả**")
    st.caption("Excel theo topic luôn chứa toàn bộ hội thoại thuộc topic, kể cả khi đang tìm kiếm hoặc lọc danh sách trên màn hình.")
    excel = create_topic_excel(selected_topic, selected, topic_info, topic_model, st.session_state.get("topic_user_labels", {}))
    st.download_button(
        f"Tải Excel của Topic {selected_topic}", data=excel,
        file_name=f"topic_{selected_topic}_hoi_thoai.xlsx", mime=EXCEL_MIME,
        on_click="ignore", key=f"step6_excel_{selected_topic}", width="stretch"
    )

    if st.button("Chuẩn bị ZIP chứa Excel của tất cả topic", key="step6_prepare_zip", width="stretch"):
        try:
            with st.spinner("Đang tạo file tổng hợp và các file topic..."):
                st.session_state["all_topics_zip_bytes"] = create_all_topics_zip(
                    topic_info, result_dataframe, topic_model, st.session_state.get("topic_user_labels", {})
                )
            st.success("Đã chuẩn bị ZIP: một file tổng hợp và Excel riêng cho từng topic.")
        except Exception as exc:
            st.session_state.pop("all_topics_zip_bytes", None)
            st.error(f"Không thể chuẩn bị file ZIP: {exc}")
    if st.session_state.get("all_topics_zip_bytes"):
        st.download_button(
            "Tải toàn bộ topic (.zip)", data=st.session_state["all_topics_zip_bytes"],
            file_name="chatbot_topics_tat_ca.zip", mime="application/zip",
            on_click="ignore", key="step6_download_zip", width="stretch", type="primary"
        )
