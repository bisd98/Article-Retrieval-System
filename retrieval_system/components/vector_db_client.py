from qdrant_client import QdrantClient, models
from qdrant_client.http import models
from tqdm.auto import tqdm


class VectorDbClient:
    def __init__(self, vector_db_path="vector_db", embedding_model=None):
        """
        Initialize a VectorDbClient object.

        Args:
            vector_db_path (str): The path to the vector database.
            embedding_model: The embedding model used for encoding text into vectors.
        """
        self.client = QdrantClient(path=vector_db_path)
        self.embedding_model = embedding_model

    def _create_collection(self, collection_name):
        """
        Create a new collection in the vector database.

        Args:
            collection_name (str): The name of the collection.
        """
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )

    def store_documents(
        self, documents, collection_name="medium-data-science-articles", batch_size=128
    ):
        """
        Store documents in the vector database.

        Args:
            documents (list): A list of documents to be stored.
            collection_name (str): The name of the collection to store the documents in.
            batch_size (int): The number of documents to upload in each batch.
        """
        num_documents = len(documents)
        num_batches = (num_documents + batch_size - 1) // batch_size

        self._create_collection(collection_name)

        for batch_idx in tqdm(range(num_batches), desc="Uploading data into vector db"):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_documents)
            batch_documents = documents[batch_start:batch_end]

            points_to_upload = [
                models.PointStruct(
                    id=idx + batch_start, vector=doc["vector"], payload=doc
                )
                for idx, doc in enumerate(batch_documents)
            ]

            self.client.upload_points(
                collection_name=collection_name, points=points_to_upload
            )

    def retrieve_documents(
        self, query_text, collection_name="medium-data-science-articles", limit=3
    ):
        """
        Retrieve documents from the vector database based on a query.

        Args:
            query_text (str): The query text used to retrieve relevant documents.
            collection_name (str): The name of the collection to retrieve documents from.
            limit (int): The maximum number of documents to retrieve.

        Returns:
            list: List of found similar documents from the vector database based on a query.
        """
        query_vector = self.embedding_model.encode(query_text).tolist()
        retrieved_documents = tqdm(
            self.client.search(
                collection_name=collection_name, query_vector=query_vector, limit=limit
            ),
            desc="Retrieving documents",
        )
        return retrieved_documents
