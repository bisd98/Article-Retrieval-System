import sys
import time
import unittest
import pandas as pd

sys.path.append('../Article-Retrieval-System/')
sys.path.append('../Article-Retrieval-System/llm/')

from llm.gpt_api import OpenAIClient
from retrieval_system.components.chunk import ArticleChunker, SentenceCountSplitterStrategy, SpacyTextSplitterStrategy
from retrieval_system.components.embedding import ArticleTransformer


class TestOpenAIClient(unittest.TestCase):
    def test_singleton_instance(self):
        """Test that only one instance of OpenAIClient is created."""
        client1 = OpenAIClient()
        client2 = OpenAIClient()
        self.assertIs(client1, client2)
        

class TestArticleChunkerWithStrategies(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Title': ['Test Article'],
            'Text': ['This is a test article content. It is supposed to be chunked into smaller parts.']
        })

    def test_chunk_text_with_spacy_strategy(self):
        """Test case for chunking text using the SpacyTextSplitterStrategy."""
        chunker = ArticleChunker(SpacyTextSplitterStrategy())
        chunks = chunker.chunk_text(self.data)
        self.assertTrue(len(chunks) > 0, "Should create at least one chunk with Spacy strategy.")
        self.assertIn('title', chunks[0], "Chunk should contain a title.")
        self.assertIn('chunk_idx', chunks[0], "Chunk should contain a chunk_idx.")
        self.assertIn('content', chunks[0], "Chunk should contain content.")

    def test_chunk_text_with_num_sentences_strategy(self):
        """
        Test the chunk_text method with the SentenceCountSplitterStrategy using a specific number of sentences.
        """
        chunker = ArticleChunker(SentenceCountSplitterStrategy(num_sentences=2))
        chunks = chunker.chunk_text(self.data)
        self.assertTrue(len(chunks) > 0, "Should create at least one chunk with Sentence Count strategy.")
        self.assertIn('title', chunks[0], "Chunk should contain a title.")
        self.assertIn('chunk_idx', chunks[0], "Chunk should contain a chunk_idx.")
        self.assertIn('content', chunks[0], "Chunk should contain content.")


class TestArticleTransformer(unittest.TestCase):
    def test_create_vectors(self):
        """Test the create_vectors method of the ArticleTransformer class."""
        transformer = ArticleTransformer(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        documents = [
            {"content": "This is a test document."},
            {"content": "This is another test document."},
        ]
        documents = transformer.create_vectors(documents)
        self.assertEqual(len(documents), 2)
        self.assertIn("vector", documents[0])
        self.assertIn("vector", documents[1])
        self.assertEqual(len(documents[0]["vector"]), 384)
        self.assertEqual(len(documents[1]["vector"]), 384)


class TestArticleTransformerPerformance(unittest.TestCase):
    def test_create_vectors_performance(self):
        """Test the performance of create_vectors method."""
        transformer = ArticleTransformer(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        documents = [{"content": "This is a test document."} for _ in range(1000)]
        start_time = time.time()
        documents = transformer.create_vectors(documents)
        end_time = time.time()
        self.assertTrue((end_time - start_time) < 60)

if __name__ == '__main__':
    unittest.main()