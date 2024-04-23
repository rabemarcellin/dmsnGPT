import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.environ.get('OPENAI_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')
