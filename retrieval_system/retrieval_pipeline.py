from retrieval_system.components import chunk, embedding, vector_db_client
import pandas as pd


class RetrieverPipeline:
    """
    A class representing the retrieval pipeline for the Article Retrieval System.

    Attributes:
        chunker (chunk.ArticleChunker): An instance of the ArticleChunker class for text chunking.
        transformer (embedding.ArticleTransformer): An instance of the ArticleTransformer class for text embedding.
        loader (vector_db_client.VectorDbClient): An instance of the VectorDbClient class for vector database operations.
        system_set (bool): A flag indicating whether the retrieval system has been set up.

    Methods:
        setup_system(path): Sets up the retrieval system by loading articles, chunking text, creating vectors, and storing them in the vector database.
        get_context(input): Retrieves relevant articles based on the input query.

    """

    def __init__(self):
        self.chunker = chunk.ArticleChunker()
        self.transformer = embedding.ArticleTransformer()
        self.loader = vector_db_client.VectorDbClient(
            embedding_model=self.transformer.emb_model
        )
        self.system_set = False

    def setup_system(self, path):
        """
        Sets up the retrieval system by loading articles, chunking text, creating vectors, and storing them in the vector database.

        Args:
            path (str): The path to the CSV file containing the articles.

        """
        articles = pd.read_csv(path)
        chunked_articles = self.chunker.chunk_text(articles)
        vector_articles = self.transformer.create_vectors(chunked_articles)

        self.loader.store_documents(vector_articles)
        self.system_set = True

    def get_context(self, input):
        """
        Retrieves relevant articles based on the input query.

        Args:
            input (str): The input query.

        Returns:
            hits (list): A list of relevant articles.

        """
        hits = self.loader.retrieve_documents(input)

        return hits


pipeline = RetrieverPipeline()
