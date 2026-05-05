"""
Cloud Function: ingest_pdf
Triggered by: google.cloud.storage.object.v1.finalized (any new file in the bucket)
Pipeline:
  1. Download the PDF from GCS into /tmp
  2. Extract text page-by-page (PyPDFLoader keeps page numbers as metadata)
  3. Split into ~1000-char chunks with 200-char overlap (same as the lab)
  4. Generate embeddings via Vertex AI text-embedding-005 (768 dims)
  5. Upsert chunks + embeddings into MongoDB Atlas (chat-rag.context)
Each chunk keeps `source` (filename) and `page` metadata so the chat backend
can cite "[file.pdf, page X]" in its answers.
"""

import os
import logging
import tempfile

import functions_framework
from cloudevents.http import CloudEvent
from google.cloud import storage

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Env vars — set at deploy time via --set-env-vars
ATLAS_URI = os.environ["ATLAS_URI"]
DB_NAME = os.environ.get("DB_NAME", "chat-rag")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "context")
INDEX_NAME = os.environ.get("INDEX_NAME", "vector_index")

# Reuse heavy clients across function invocations (warm starts)
_mongo_client = MongoClient(ATLAS_URI)
_collection = _mongo_client[DB_NAME][COLLECTION_NAME]
_embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
_storage_client = storage.Client()


@functions_framework.cloud_event
def ingest_pdf(cloud_event: CloudEvent) -> None:
    data = cloud_event.data
    bucket_name = data["bucket"]
    object_name = data["name"]

    # Skip anything that's not a PDF (e.g. folder placeholders, other formats)
    if not object_name.lower().endswith(".pdf"):
        log.info("Skipping non-PDF object: %s", object_name)
        return

    log.info("Ingesting gs://%s/%s", bucket_name, object_name)

    # 1. Download the PDF locally
    bucket = _storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        local_path = tmp.name
    blob.download_to_filename(local_path)

    try:
        # 2. Extract text — one Document per page
        loader = PyPDFLoader(local_path)
        pages = loader.load()
        log.info("Extracted %d pages from %s", len(pages), object_name)

        # 3. Split into chunks (same params as the lab)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(pages)

        # Stamp each chunk with the source filename so we can cite it later.
        # PyPDFLoader already adds metadata["page"] (0-indexed).
        for chunk in chunks:
            chunk.metadata["source"] = object_name
            chunk.metadata["bucket"] = bucket_name

        log.info("Split into %d chunks", len(chunks))

        # 4 + 5. Embed and upsert into Atlas
        vector_store = MongoDBAtlasVectorSearch(
            collection=_collection,
            embedding=_embeddings,
            index_name=INDEX_NAME,
            text_key="text",
            embedding_key="embedding",
        )
        ids = vector_store.add_documents(chunks)
        log.info("Upserted %d chunks into %s.%s", len(ids), DB_NAME, COLLECTION_NAME)

    finally:
        os.unlink(local_path)
