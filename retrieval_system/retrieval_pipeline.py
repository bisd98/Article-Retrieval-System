from retrieval_system.components import chunk, embedding, vector_db_client
import pandas as pd


class RetrieverPipeline():
    
    def __init__(self):
        self.chunker = chunk.ArticleChunker()
        self.transformer = embedding.ArticleTransformer()
        self.loader = vector_db_client.VectorDbClient(embedding_model=self.transformer.emb_model)
    
    def setup_system(self, path):
        articles = pd.read_csv(path)
        chunked_articles = self.chunker.chunk_text(articles)
        vector_articles = self.transformer.create_vectors(chunked_articles)
        
        self.loader.store_documents(vector_articles)

    def get_context(self, input):
        hits = self.loader.retrieve_documents(input)
        
        return hits

pipeline = RetrieverPipeline()