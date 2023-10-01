import logging
from abc import ABC, abstractmethod

import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone.core.client.exceptions import ApiException
from tqdm.auto import tqdm

_TEXT_FIELD = "text"

logger = logging.getLogger(__name__)


class IndexBuilder(ABC):
    def __init__(self, index_name, pinecone_api_key, pinecone_env, openai_api_key):
        self.index_name = index_name
        self.embeddings = self.get_embeddings(openai_api_key)
        self.dim = self.get_embeddings_dim(self.embeddings)
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.openai_api_key = openai_api_key

    @staticmethod
    def get_embeddings_dim(embeddings):
        res = embeddings.embed_documents("sample text")
        return len(res[0])

    def get_embeddings(self, openai_api_key):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        dim = self.get_embeddings_dim(embeddings)
        return embeddings

    def initialize_pinecone(self):
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name, metric="cosine", dimension=self.dim
            )
        else:
            print(f"Pinecone index already exist: {self.index_name}")
        return pinecone.Index(self.index_name)

    def upload_to_pinecone(self, db, index):
        print(f"Uploading to Pineline namespace={self.namespace}")
        chroma_result_fields = ["metadatas", "documents", "embeddings"]
        num_embeds = db._collection.count()
        batch_size = 16

        for i in tqdm(range(0, num_embeds, batch_size)):
            # set end position of batch
            i_end = min(i + batch_size, num_embeds)

            result = db.get(offset=i, limit=batch_size, include=chroma_result_fields)
            lines_batch = result["documents"]
            ids_batch = result["ids"]
            embeds = result["embeddings"]
            metadata = result["metadatas"]
            for j, line in enumerate(lines_batch):
                metadata[j][_TEXT_FIELD] = line
            to_upsert = zip(ids_batch, embeds, metadata)
            try:
                index.upsert(vectors=list(to_upsert), namespace=self.namespace)
            except ApiException as e:
                logger.error(e)

    @abstractmethod
    def load_documents(self):
        pass

    @property
    @abstractmethod
    def namespace(self):
        pass

    @abstractmethod
    def create_db(self):
        pass

    def run(self):
        self.load_documents()
        db = self.create_db()
        index = self.initialize_pinecone()
        self.upload_to_pinecone(db, index)
