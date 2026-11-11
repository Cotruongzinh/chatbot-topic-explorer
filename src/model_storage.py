
# Luu/tai model BERTopic local. Pickle chi duoc tai tu nguon tin cay.
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from bertopic import BERTopic

from .bertopic_engine import MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
METADATA_FILE = MODEL_DIR / "bertopic_model_metadata.json"
SCHEMA_VERSION = 1


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def get_saved_model_metadata():
    if not METADATA_FILE.is_file():
        return None
    try:
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return metadata if isinstance(metadata, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _model_path_from_metadata(metadata: dict) -> Path:
    filename = metadata.get("model_file")
    if not filename or Path(filename).name != filename:
        raise ValueError("Ten file model trong metadata khong hop le.")
    model_path = (MODEL_DIR / filename).resolve()
    if model_path.parent != MODEL_DIR.resolve():
        raise ValueError("Duong dan model nam ngoai thu muc models.")
    return model_path


def _is_compatible(metadata, analysis_signature, document_count) -> bool:
    if not isinstance(metadata, dict):
        return False
    if metadata.get("schema_version") != SCHEMA_VERSION:
        return False
    if metadata.get("analysis_signature") != analysis_signature:
        return False
    try:
        if int(metadata.get("document_count", -1)) != int(document_count):
            return False
    except (TypeError, ValueError):
        return False
    if metadata.get("bertopic_version") != _package_version("bertopic"):
        return False
    saved_python = str(metadata.get("python_major_minor", ""))
    current_python = ".".join(platform.python_version().split(".")[:2])
    if saved_python != current_python:
        return False
    try:
        model_path = _model_path_from_metadata(metadata)
    except (TypeError, ValueError):
        return False
    return model_path.is_file()


def is_saved_model_compatible(analysis_signature, document_count) -> bool:
    return _is_compatible(
        get_saved_model_metadata(),
        analysis_signature,
        document_count,
    )


def _normalise_label_map(labels) -> dict:
    result = {}
    if not isinstance(labels, dict):
        return result
    for key, value in labels.items():
        try:
            topic_id = int(key)
        except (TypeError, ValueError):
            continue
        if value is not None:
            result[str(topic_id)] = str(value)
    return result


def _dataframe_records(dataframe):
    if dataframe is None:
        return None
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("topic_info_before_refinement phai la pandas DataFrame.")
    return json.loads(dataframe.to_json(orient="records", force_ascii=False))


def save_topic_model(
    topic_model,
    analysis_signature,
    text_column,
    source_filename,
    document_count,
    topic_auto_labels=None,
    topic_user_labels=None,
    vn_keywords_refined=False,
    topic_info_before_refinement=None,
):
    topics = getattr(topic_model, "topics_", None)
    if topics is None:
        raise ValueError("Model chua duoc fit, khong the luu.")
    if len(topics) != int(document_count):
        raise ValueError(
            f"So topic ({len(topics)}) khong khop so van ban ({document_count})."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    model_filename = f"bertopic_model_{timestamp}.pkl"
    final_model_path = MODEL_DIR / model_filename
    temporary_model_path = MODEL_DIR / f".{model_filename}.tmp"
    temporary_metadata_path = MODEL_DIR / ".bertopic_model_metadata.tmp"
    metadata_committed = False

    vectorizer = getattr(topic_model, "vectorizer_model", None)
    missing = object()
    previous_stop_words = getattr(vectorizer, "stop_words_", missing)
    previous_embedding_model = getattr(topic_model, "embedding_model", None)

    try:
        try:
            topic_model.save(
                str(temporary_model_path),
                serialization="pickle",
                save_embedding_model=False,
            )
        finally:
            # BERTopic co the tam thay embedding_model va stop_words_ khi pickle.
            topic_model.embedding_model = previous_embedding_model
            if vectorizer is not None and previous_stop_words is not missing:
                vectorizer.stop_words_ = previous_stop_words

        if not temporary_model_path.is_file() or temporary_model_path.stat().st_size == 0:
            raise RuntimeError("BERTopic khong tao duoc file model.")

        topic_ids = [int(topic) for topic in topics]
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "serialization": "pickle",
            "model_file": model_filename,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "analysis_signature": str(analysis_signature),
            "text_column": str(text_column),
            "source_filename": str(source_filename),
            "document_count": int(document_count),
            "topic_count": len(set(topic_ids) - {-1}),
            "outlier_count": sum(topic == -1 for topic in topic_ids),
            "topic_auto_labels": _normalise_label_map(topic_auto_labels),
            "topic_user_labels": _normalise_label_map(topic_user_labels),
            "vn_keywords_refined": bool(vn_keywords_refined),
            "topic_info_before_refinement": _dataframe_records(
                topic_info_before_refinement
            ),
            "embedding_model": MODEL_NAME,
            "python_version": platform.python_version(),
            "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
            "bertopic_version": _package_version("bertopic"),
        }

        with temporary_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        # Commit metadata sau cung de metadata cu van tro den model cu neu loi.
        os.replace(temporary_model_path, final_model_path)
        os.replace(temporary_metadata_path, METADATA_FILE)
        metadata_committed = True

        # Chi giu model moi nhat de thu muc models khong phinh to.
        for old_model in MODEL_DIR.glob("bertopic_model_*.pkl"):
            if old_model != final_model_path:
                try:
                    old_model.unlink()
                except OSError:
                    pass

        return metadata

    finally:
        if not metadata_committed and final_model_path.exists():
            try:
                final_model_path.unlink()
            except OSError:
                pass
        for temporary_path in (temporary_model_path, temporary_metadata_path):
            try:
                if temporary_path.exists():
                    temporary_path.unlink()
            except OSError:
                pass


def load_topic_model(
    embedding_model,
    expected_analysis_signature,
    expected_document_count,
):
    metadata = get_saved_model_metadata()
    if not _is_compatible(
        metadata,
        expected_analysis_signature,
        expected_document_count,
    ):
        raise ValueError(
            "Model da luu khong tuong thich voi file CSV/cot van ban hien tai "
            "hoac khac phien ban BERTopic/Python."
        )

    model_path = _model_path_from_metadata(metadata)
    topic_model = BERTopic.load(
        str(model_path),
        embedding_model=embedding_model,
    )
    if getattr(topic_model, "topics_", None) is None:
        raise RuntimeError("File da tai khong chua fitted topics.")
    return topic_model, metadata
