import os
import configparser
from langchain.document_loaders import PyPDFLoader
import logging
import uuid
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from openai.error import APIError
import pinecone
from tqdm.auto import tqdm
from pinecone.core.client.exceptions import ApiException
import time
import argparse

from index_builder import IndexBuilder

logger = logging.getLogger(__name__)

class BookIndexBuilder(IndexBuilder):
    def __init__(self, index_name, pinecone_api_key, pinecone_env, openai_api_key, filename):
        super().__init__(index_name, pinecone_api_key, pinecone_env, openai_api_key)
        self.filename = filename
        self._namespace = str(uuid.uuid4())

    def load_documents(self):
        print("Loading PDF...")
        loader = PyPDFLoader(self.filename)
        self.pages = loader.load_and_split()
        print("Pages:")
        print(len(self.pages))
        print(self.pages[0])

    @property
    def namespace(self):
        return self._namespace

    def create_db(self):
        return Chroma.from_documents(self.pages, self.embeddings)

def main():
    parser = argparse.ArgumentParser(description='Builds an index for a given PDF file.')
    parser.add_argument('filename', type=str, help='Path to the PDF file')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')

    indexer = BookIndexBuilder(config.get('settings', 'pinecone_index'), config.get('api', 'pinecone_api_key'), config.get('api', 'pinecone_env'), config.get('api', 'openai_api_key'), args.filename)
    indexer.run()

if __name__ == "__main__":
    main()
