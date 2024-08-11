# Generator function for returning the data a record at a time.
from typing import Generator

import polars as pl


def stream_docs(path: str) -> Generator[tuple[str, int], None, None]:
    """
    Generator function that streams documents from a Parquet file.

    Parameters
    ----------
    path : str
        The path to the Parquet file.

    Yields
    ------
    tuple[str, int]
        A tuple containing the review text and sentiment label.
    """
    # Create a lazy frame and collect the data in batches
    l_frame: pl.LazyFrame = pl.scan_parquet(path)
    for row in l_frame.collect().iter_rows(named=True):
        text: str = row["review"]
        label: int = int(row["sentiment"])
        yield text, label


def get_minibatch(
    doc_stream: Generator[tuple[str, int], None, None], size: int
) -> tuple[list[str] | None, list[int] | None]:
    """
    Retrieve a minibatch of documents and labels from the document stream.

    Parameters
    ----------
    doc_stream : Generator[tuple[str, int], None, None]
        A generator that yields tuples of (text, label).
    size : int
        The number of documents to retrieve for the minibatch.

    Returns
    -------
    tuple[list[str] | None, list[int] | None]
        A tuple containing two lists: one for documents and one for labels.
        Returns (None, None) if the stream is exhausted.
    """
    docs: list[str] = []
    labels: list[int] = []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            labels.append(label)
    except StopIteration:
        return None, None

    return docs, labels
