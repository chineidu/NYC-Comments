import os
from collections.abc import Iterable
from typing import Callable, Iterator, List, Literal, Tuple

import spacy
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, Phrases
from matplotlib import pyplot as plt
from rich.console import Console
from rich.theme import Theme
from smart_open import open
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
from wordcloud import WordCloud

custom_theme = Theme({"info": "#76FF7B", "warning": "#FBDDFE", "error": "#FF0000"})
console = Console(theme=custom_theme)


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

    def __len__(self) -> int:
        return sum(1 for _ in open(self.filepath, "r"))

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


class LdaTopicExtractor:
    """
    A class for performing topic modeling using LDA.

    Steps required to perform topic modeling:
    1. Create a BoW corpus using the corpus_iterable / preprocessed corpus.
    2. Create n-grams from the corpus.
    3. Create an LDA model using the corpus and TF-IDF representation.
    4. Use the LDA model to extract topics and their corresponding words.

    Attributes
    ----------
    corpus : Iterable[List[str]]
        The input corpus for topic modeling.
    num_topics : int
        The number of topics to extract.
    chunksize : int
        The number of documents to be used in each training chunk.
    iterations : int
        Maximum number of iterations through the corpus when inferring the topic
        distribution of a corpus.
    passes : int
        Number of passes through the corpus during training.
    """

    def __init__(
        self,
        corpus: Iterable[List[str]],
        num_topics: int = 10,
        chunksize: int = 2_000,
        iterations: int = 400,
        passes: int = 30,
    ) -> None:
        """
        Initialize the TopicAnalyzer.

        Parameters
        ----------
        corpus : Iterable[List[str]]
            The input corpus for topic modeling.
        num_topics : int, optional
            The number of topics to extract, by default 10.
        chunksize : int, optional
            The number of documents to be used in each training chunk, by default 2000.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution
            of a corpus, by default 400.
        passes : int, optional
            Number of passes through the corpus during training, by default 30.
        """
        self.corpus: Iterable[List[str]] = corpus
        self.num_topics: int = num_topics
        self.chunksize: int = chunksize
        self.iterations: int = iterations
        self.passes: int = passes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_corpus={len(self.corpus):,})"  # type: ignore

    def _create_bigrams(self) -> Phrases:
        """
        Create bigrams from the corpus.

        Returns
        -------
        Phrases
            A Phrases model for creating bigrams.
        """
        return Phrases(sentences=self.corpus, min_count=20)

    def add_bigrams(self) -> Iterable[List[str]]:
        """
        Add bigrams to the corpus.

        Returns
        -------
        Iterable[List[str]]
            The corpus with added bigrams.
        """
        bigram: Phrases = self._create_bigrams()
        for doc in self.corpus:
            yield (doc + [token for token in bigram[doc] if "_" in token])

    def create_bow(self) -> Tuple[Dictionary, List[List[Tuple[int, int]]]]:
        """
        Create a Bag of Words (BoW) representation of the corpus.

        Returns
        -------
        Tuple[Dictionary, List[List[Tuple[int, int]]]]
            A tuple containing the dictionary and the BoW corpus.
        """
        self.corpus_wth_bigrams_: List[List[str]] = list(self.add_bigrams())
        dictionary_: Dictionary = Dictionary(self.corpus_wth_bigrams_)
        corpus_bow: List[List[Tuple[int, int]]] = [
            dictionary_.doc2bow(doc) for doc in self.corpus_wth_bigrams_
        ]
        return dictionary_, corpus_bow

    def train_lda(self) -> LdaModel:
        """
        Train the LDA model.

        Returns
        -------
        LdaModel
            The trained LDA model.
        """
        dictionary_: Dictionary
        corpus_bow: List[List[Tuple[int, int]]]
        dictionary_, corpus_bow = self.create_bow()
        self.lda_model: LdaModel = LdaModel(
            corpus=corpus_bow,
            id2word=dictionary_,
            chunksize=self.chunksize,
            alpha="auto",
            eta="auto",
            iterations=self.iterations,
            num_topics=self.num_topics,
            passes=self.passes,
            eval_every=None,
        )
        console.print("LDA model trained.", style="info")
        return self.lda_model

    def extract_topics(self) -> List[Tuple[List[Tuple[str, float]], float]]:
        """
        Extract topics from the trained LDA model.

        Returns
        -------
        List[Tuple[List[Tuple[str, float]], float]]
            A list of tuples containing the top topics and their coherence scores.
        """
        corpus_bow: List[List[Tuple[int, int]]]
        _, corpus_bow = self.create_bow()
        self.top_topics: List[Tuple[List[Tuple[str, float]], float]] = self.lda_model.top_topics(
            corpus=corpus_bow, topn=self.num_topics
        )
        avg_topic_coherence: float = sum([t[1] for t in self.top_topics]) / self.num_topics
        console.print(f"Average topic coherence: {avg_topic_coherence:.4f}\n", style="info")
        return self.top_topics

    @staticmethod
    def print_topics(top_topics: List[Tuple[List[Tuple[str, float]], float]]) -> None:
        """
        Print the top topics.

        Parameters
        ----------
        top_topics : List[Tuple[List[Tuple[str, float]], float]]
            A list of tuples containing the top topics and their coherence scores.
        """
        console.print(top_topics, style="info")

    @staticmethod
    def _compute_topic_coherence(
        model: LdaModel, corpus_with_bigrams: List[List[str]], dictionary: Dictionary
    ) -> float:
        """
        Compute the topic coherence score.

        Parameters
        ----------
        model : LdaModel
            The trained LDA model.
        corpus_with_bigrams : List[List[str]]
            The corpus with bigrams.
        dictionary : Dictionary
            The dictionary used for creating the BoW corpus.

        Returns
        -------
        float
            The computed coherence score.
        """
        coherence_model: CoherenceModel = CoherenceModel(
            model=model,
            texts=corpus_with_bigrams,
            dictionary=dictionary,
            coherence="u_mass",
        )
        coherence_score: float = coherence_model.get_coherence()
        return coherence_score

    @staticmethod
    def _compute_perplexity(model: LdaModel, corpus_bow: List[List[Tuple[int, int]]]) -> float:
        """
        Compute the perplexity of the model.

        Parameters
        ----------
        model : LdaModel
            The trained LDA model.
        corpus_bow : List[List[Tuple[int, int]]]
            The BoW representation of the corpus.

        Returns
        -------
        float
            The computed perplexity.
        """
        perplexity: float = model.log_perplexity(corpus_bow)
        return perplexity

    def evaluate_lda(self) -> Tuple[float, float]:
        """
        Evaluate the LDA model by computing coherence score and perplexity.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the coherence score and perplexity.
        """
        dictionary_: Dictionary
        corpus_bow: List[List[Tuple[int, int]]]
        dictionary_, corpus_bow = self.create_bow()
        model: LdaModel = self.train_lda()
        coherence_score: float = self._compute_topic_coherence(
            model, self.corpus_wth_bigrams_, dictionary_
        )
        perplexity: float = self._compute_perplexity(model, corpus_bow)
        console.print(
            f"N_topics: {self.num_topics} | Coherence score: {coherence_score:.4f} "
            f"| Perplexity: {perplexity:.4f}",
            style="info",
        )
        return coherence_score, perplexity
