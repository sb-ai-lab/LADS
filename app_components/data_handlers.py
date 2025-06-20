import pandas as pd
import streamlit as st
from io import StringIO, BytesIO

SUPPORTED_FILE_TYPES = {
    'csv': (pd.read_csv, lambda df, path: df.to_csv(path)),
    'xlsx': (pd.read_excel, lambda df, path: df.to_excel(path)),
    'xls': (pd.read_excel, lambda df, path: df.to_excel(path)),
    'parquet': (pd.read_parquet, lambda df, path: df.to_parquet(path))
}


def load_data(file_content: bytes, file_extension: str) -> pd.DataFrame:

    if file_extension.lower() not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file type: {file_extension.lower()}. "
            f"Supported types are {', '.join(SUPPORTED_FILE_TYPES.keys())}."
        )

    try:
        reader_func, _ = SUPPORTED_FILE_TYPES[file_extension.lower()]
        if file_extension.lower() == 'csv':
            df = reader_func(StringIO(file_content.decode("utf-8")), index_col=0)
        else:
            df = reader_func(BytesIO(file_content))
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise


def save_file_to_disk(df: pd.DataFrame, file_name: str, file_extension: str) -> None:
    try:
        _, writer_func = SUPPORTED_FILE_TYPES[file_extension.lower()]
        writer_func(df, './datasets/'+file_name)
    except Exception as e:
        st.error(f"Error saving file to disk: {str(e)}")
