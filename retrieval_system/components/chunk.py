from langchain.text_splitter import SpacyTextSplitter
import spacy.cli
from tqdm.auto import tqdm
import spacy
import logging
import warnings

logging.disable()
spacy.cli.download("en_core_web_lg")


class ArticleChunker:
    def __init__(
        self,
        pipeline="en_core_web_lg",
        chunk_size=512,
        chunk_overlap=64,
        batch_size=256,
    ):
        """
        Initializes an ArticleChunker object.

        Args:
            pipeline (str, optional): The name of the spaCy pipeline to use for text processing. Defaults to "en_core_web_lg".
            chunk_size (int, optional): The size of each chunk in characters. Defaults to 512.
            chunk_overlap (int, optional): The number of characters to overlap between adjacent chunks. Defaults to 64.
            batch_size (int, optional): The number of samples to process in each batch. Defaults to 256.
        """
        self.splitter = SpacyTextSplitter(
            pipeline=pipeline, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.nlp = spacy.load(pipeline)
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
                chunks = self.splitter.split_text("".join(row.Text.split("\n\n")))
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        {"title": row.Title, "chunk_idx": idx, "content": chunk}
                    )

        return documents
