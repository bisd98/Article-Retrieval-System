from abc import ABC, abstractmethod
from langchain.text_splitter import SpacyTextSplitter
import spacy.cli
from tqdm.auto import tqdm
import spacy
import logging
import warnings

logging.disable()

try:
    spacy.load("en_core_web_lg")
except:
    spacy.cli.download("en_core_web_lg")

class TextSplitterStrategy(ABC):
    @abstractmethod
    def split_text(self, text):
        """
        Split the given text into chunks.

        Args:
            text (str): The text to be split.

        Returns:
            list: A list of text chunks.
        """
        pass

class SpacyTextSplitterStrategy(TextSplitterStrategy):
    def __init__(self, pipeline="en_core_web_lg", chunk_size=512, chunk_overlap=64):
        """
        Initializes a SpacyTextSplitterStrategy object.

        Args:
            pipeline (str, optional): The name of the spaCy pipeline to use for text processing. Defaults to "en_core_web_lg".
            chunk_size (int, optional): The maximum size of each chunk. Defaults to 512.
            chunk_overlap (int, optional): The overlap size between chunks. Defaults to 64.
        """
        self.splitter = SpacyTextSplitter(
            pipeline=pipeline, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_text(self, text):
        """
        Split the given text into chunks using SpacyTextSplitter.

        Args:
            text (str): The text to be split.

        Returns:
            list: A list of text chunks.
        """
        return self.splitter.split_text(text)

class SentenceCountSplitterStrategy(TextSplitterStrategy):
    def __init__(self, num_sentences=5):
        """
        Initializes a SentenceCountSplitterStrategy object.

        Args:
            num_sentences (int, optional): The number of sentences per chunk. Defaults to 5.
        """
        self.num_sentences = num_sentences

    def split_text(self, text):
        """
        Split the given text into chunks based on the number of sentences.

        Args:
            text (str): The text to be split.

        Returns:
            list: A list of text chunks.
        """
        doc = spacy.load("en_core_web_lg")(text)
        sentences = [sent.text for sent in doc.sents]
        chunks = [
            " ".join(sentences[i:i + self.num_sentences])
            for i in range(0, len(sentences), self.num_sentences)
        ]
        return chunks

class ArticleChunker:
    def __init__(self, split_strategy=None, batch_size=128):
        """
        Initializes an ArticleChunker object.

        Args:
            split_strategy (TextSplitterStrategy, optional): The strategy to use for text splitting. 
            If not provided, SpacyTextSplitterStrategy will be used as default.
            batch_size (int, optional): The number of samples to process in each batch. Defaults to 128.
        """
        if split_strategy is None:
            split_strategy = SpacyTextSplitterStrategy()
        self.split_strategy = split_strategy
        self.batch_size = batch_size

    def chunk_text(self, data):
        """
        Chunk the given text data into smaller chunks.

        Args:
            data (pandas.DataFrame): The input data containing the text to be chunked.

        Returns:
            list: A list of dictionaries, where each dictionary represents a chunk and contains the following keys:
                - "title": The title of the article.
                - "chunk_idx": The index of the chunk within the article.
                - "content": The content of the chunk.
        """
        documents = []
        num_samples = len(data)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        warnings.simplefilter("ignore")

        for batch_idx in tqdm(range(num_batches), desc="Chunking articles"):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, num_samples)
            batch_data = data.iloc[batch_start:batch_end]

            for _, row in batch_data.iterrows():
                chunks = self.split_strategy.split_text("".join(row.Text.split("\n\n")))
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        {"title": row.Title, "chunk_idx": idx, "content": chunk}
                    )

        return documents
