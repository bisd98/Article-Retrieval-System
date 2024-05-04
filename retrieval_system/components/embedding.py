from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm


class ArticleTransformer:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.emb_model = SentenceTransformer(model_name, device=device)

    def create_vectors(self, documents, batch_size=512):
        num_documents = len(documents)
        num_batches = (num_documents + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(num_batches), desc=f"Processing batches with {self.emb_model.device}"
        ):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_documents)
            batch_docs = documents[batch_start:batch_end]

            embeddings = self.emb_model.encode([doc["content"] for doc in batch_docs])

            for idx, emb in enumerate(embeddings):
                batch_docs[idx]["vector"] = emb.tolist()

        return documents
