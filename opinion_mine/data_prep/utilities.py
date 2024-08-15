import os
from typing import Callable, Iterator, List, Literal

import spacy
from matplotlib import pyplot as plt
from smart_open import open
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
from wordcloud import WordCloud


def tokenizer_spacy(
    nlp: spacy.language.Language,
    corpus: list[str],
    batch_size: int = 1_000,
    n_process: Literal[1, 2, 4, -1] = 2,
    drop_stopwords: bool = True,
) -> list[list[str]]:
    """
    Tokenize a corpus of text using spaCy.

    Parameters
    ----------
    nlp : spacy.language.Language
        The spaCy language model to use for tokenization.
    corpus : list[str]
        A list of strings to tokenize. Shape: (n_documents,)
    batch_size : int, optional
        The number of texts to process in each batch (default is 1000).
    n_process : Literal[1, 2, 4, -1], optional
        The number of processes to use for parallel processing (default is 2).
    drop_stopwords : bool, optional
        Whether to remove stop words from the tokens (default is True).

    Returns
    -------
    list[list[str]]
        A list of tokenized documents, where each document is represented as a list of strings.
        Shape: (n_documents, n_tokens_per_document)

    Raises
    ------
    ValueError
        If n_process is not one of the allowed values [1, 2, 4, -1].
    """

    if n_process not in [1, 2, 4, -1]:
        raise ValueError(f"n_process must be in [1, 2, 4, -1], got {n_process}")

    tokens: list[list[str]] = []
    # Filtering function
    if drop_stopwords:
        filter_fn: Callable[[Token], bool] = (
            lambda token: token.text.isalnum()
            and token.text.lower() not in STOP_WORDS
            and len(token.text) > 1
        )
    else:
        filter_fn = lambda token: token.text.isalnum() and len(token.text) > 1

    # Tokenize the corpus
    for doc in nlp.pipe(corpus, batch_size=batch_size, n_process=n_process):
        tokens.append([token.lemma_.lower() for token in doc if filter_fn(token)])

    return tokens


def save_tokenized_corpus(tok_corpus: list[list[str]], filepath: str, separator: str) -> None:
    """
    Save a tokenized corpus to a file.

    Parameters
    ----------
    tok_corpus : list[list[str]]
        The tokenized corpus to save. Shape: (n_documents, n_tokens_per_document)
    filepath : str
        The path to the file where the corpus will be saved.
    separator : str
        The separator to use between tokens in each document.

    Returns
    -------
    None

    Notes
    -----
    If the file already exists, the function will not overwrite it and will
    print a message indicating that the save operation was skipped.
    """
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            for doc in tok_corpus:
                f.write(separator.join(doc) + "\n")
        print(f"Saved tokenized corpus to {filepath}")  # noqa: T201
        return
    print(f"File {filepath!r} already exists. Skipping save operation.")  # noqa: T201


class MyCorpus:
    """
    A class to represent a corpus of text data.

    Parameters
    ----------
    filepath : str
        The path to the file containing the corpus data.
    separator : str, optional
        The separator used in the file to separate tokens (default is ",").

    Attributes
    ----------
    filepath : str
        The path to the file containing the corpus data.
    separator : str
        The separator used in the file to separate tokens.
    """

    def __init__(self, filepath: str, separator: str = ",") -> None:
        self.filepath: str = filepath
        self.separator: str = separator

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r})"

    def __iter__(self) -> Iterator[List[str]]:
        """
        Iterate over the corpus, yielding tokenized lines.

        Yields
        ------
        List[str]
            A list of tokens from each line in the corpus.
        """
        for line in open(self.filepath, "r"):
            # Corpus has already been converted to lowercase
            yield line.strip().split(self.separator)


def stream_corpus(corpus_iterable: MyCorpus | list[list[str]], size: int = 100) -> list[list[str]]:
    """
    Stream a corpus of tokens from an iterable source.

    Parameters
    ----------
    corpus_iterable : MyCorpus | list[list[str]]
        An iterable containing tokenized documents. Each document is a list of strings.
    size : int, optional
        The number of documents to stream from the corpus, by default 100.

    Returns
    -------
    list[list[str]]
        A list of tokenized documents, where each document is a list of strings.
        Shape: (n_documents, n_tokens_per_document)

    Notes
    -----
    If the corpus is smaller than the specified size, it will return all available documents
    and print a message indicating that the corpus is smaller than the requested size.
    """
    all_tokens: list[list[str]] = []
    try:
        for _ in range(size):
            tokens: list[str] = next(iter(corpus_iterable))
            all_tokens.append(tokens)
    except StopIteration:
        print(f"Corpus is smaller than the size {size}")  # noqa: T201
    return all_tokens


def create_wordcloud(topic_words: dict[str, int], title: str) -> None:
    """
    Create and display a word cloud from topic words.

    Parameters
    ----------
    topic_words : dict[str, int]
        A dictionary of words and their frequencies.
        Shape: (n_words,)
    title : str
        The title for the word cloud plot.

    Returns
    -------
    None
        This function displays the word cloud plot but does not return any value.

    Notes
    -----
    This function uses matplotlib to display the word cloud.
    """
    wordcloud: WordCloud = WordCloud(
        width=800, height=500, background_color="white"
    ).generate_from_frequencies(topic_words)

    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title.title(), fontsize=16)
    plt.tight_layout()
    plt.show()
