from io import BytesIO

import pandas as pd


def read_csv_file(uploaded_file):
    """Đọc CSV, tự nhận diện dấu phân cách và encoding."""

    raw_data = uploaded_file.getvalue()

    if not raw_data:
        raise ValueError("File CSV không có dữ liệu.")

    dataframe = None
    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "cp1258"):
        try:
            dataframe = pd.read_csv(
                BytesIO(raw_data),
                sep=None,
                engine="python",
                encoding=encoding
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            last_error = error

    if dataframe is None:
        raise ValueError(
            f"Không thể đọc file CSV. Chi tiết: {last_error}"
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if dataframe.columns.duplicated().any():
        raise ValueError("File CSV có tên cột bị trùng.")

    if len(dataframe.columns) == 0:
        raise ValueError("File CSV không có cột dữ liệu.")

    if len(dataframe) == 0:
        raise ValueError("File CSV không có bản ghi.")

    return dataframe


def get_text_columns(dataframe):
    """Tìm các cột có khả năng chứa nội dung văn bản."""

    candidates = []

    for column in dataframe.columns:
        series = dataframe[column].dropna().astype(str)

        if series.empty:
            continue

        average_length = series.str.len().mean()
        max_length = series.str.len().max()

        if average_length >= 20 or max_length >= 80:
            candidates.append(column)

    if not candidates:
        candidates = [
            column
            for column in dataframe.columns
            if (
                pd.api.types.is_object_dtype(dataframe[column])
                or pd.api.types.is_string_dtype(dataframe[column])
            )
        ]

    return candidates


def get_data_quality(dataframe, text_column):
    """Thống kê chất lượng dữ liệu."""

    text_series = (
        dataframe[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    duplicate_ids = None

    if "ma_hoi_thoai" in dataframe.columns:
        ids = dataframe["ma_hoi_thoai"].dropna().astype(str)
        duplicate_ids = int(ids.duplicated().sum())

    return {
        "total_rows": len(dataframe),
        "total_columns": len(dataframe.columns),
        "missing_cells": int(dataframe.isna().sum().sum()),
        "empty_text": int(text_series.eq("").sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "duplicate_text": int(
            text_series[text_series.ne("")].duplicated().sum()
        ),
        "duplicate_ids": duplicate_ids
    }


def prepare_documents(dataframe, text_column):
    """Lấy văn bản hợp lệ và giữ lại metadata tương ứng."""

    text_series = (
        dataframe[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    valid_mask = text_series.ne("")

    documents = text_series[valid_mask].tolist()

    valid_dataframe = (
        dataframe.loc[valid_mask]
        .copy()
        .reset_index(drop=True)
    )

    if not documents:
        raise ValueError("Không tìm thấy nội dung hợp lệ để phân tích.")

    return documents, valid_dataframe
