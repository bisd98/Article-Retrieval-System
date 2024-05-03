from langchain.text_splitter import SpacyTextSplitter
from tqdm.auto import tqdm
import spacy
import logging
import warnings

logging.disable()

class ArticleChunker:
    def __init__(self, pipeline='en_core_web_lg', chunk_size=512, chunk_overlap=64, batch_size=256):
        self.splitter = SpacyTextSplitter(pipeline=pipeline, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.nlp = spacy.load(pipeline, disable=['tagger', 'attribute_ruler'])
        self.batch_size = batch_size

    def chunk_text(self, data):
        documents = []
        num_samples = len(data)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        warnings.simplefilter("ignore")

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, num_samples)
            batch_data = data.iloc[batch_start:batch_end]

            for _, row in batch_data.iterrows():
                chunks = self.splitter.split_text("".join(row.Text.split("\n\n")))
                for idx, chunk in enumerate(chunks):
                    documents.append({'title': row.Title, 'chunk_idx': idx, 'content': chunk})

        return documents