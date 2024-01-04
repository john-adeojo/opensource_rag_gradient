from llama_index import VectorStoreIndex, ServiceContext
# from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser import TokenTextSplitter
from llama_index.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM
from llama_index import StorageContext
from llama_index import load_index_from_storage
import json
# from typing import Any, List
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
import wikipedia
import json
import os
import setup
from llama_index import set_global_service_context
import nest_asyncio
nest_asyncio.apply() 

setup.set_environment_variables()

config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configurations.json')
with open(config_file_path, 'r') as file:
    config = json.load(file)
index_path = config['file_paths']['index_file']

# Functions for creating Wikipedia Index
# def load_index(filepath: str):
#     storage_context = StorageContext.from_defaults(persist_dir=index_path)
#     return load_index_from_storage(storage_context)

def load_index(service_context):
    config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configurations.json')
    print("config_file_path", config_file_path)
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    index_path = config['file_paths']['index_file']
    print("index_path!!!", index_path)

    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    return load_index_from_storage(storage_context=storage_context, service_context=service_context)


def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_wikidocs(wikipage_requests):
    list_of_wikipages = [wikipage_requests]
    print(f"Preparing to Download:{list_of_wikipages}")
    documents = []
    for page_title in list_of_wikipages:
        try:
            wiki_page = wikipedia.page(page_title)
            page_content = wiki_page.content
            page_url = wiki_page.url
            document = Document(text=page_content, metadata={'source_url': page_url})
            documents.append(document)
        except wikipedia.exceptions.PageError:
            print(f"PageError: The page titled '{page_title}' does not exist on Wikipedia.")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"DisambiguationError: The page titled '{page_title}' is ambiguous. Possible options: {e.options}")
    print("Finished downloading pages")
    return documents

def index_wikipedia_pages(wikipage_requests):
    print(f"Preparing to index Wikipages: {wikipage_requests}")
    
    documents = create_wikidocs(wikipage_requests)
    parser = TokenTextSplitter(chunk_size=150, chunk_overlap=45)
    
    embed_model = GradientEmbedding(
    gradient_access_token=os.environ['GRADIENT_ACCESS_TOKEN'],
    gradient_workspace_id=os.environ['GRADIENT_WORKSPACE_ID'],
    gradient_model_slug="bge-large",
    )

    # Use the open source LLMs hosted by Gradient
    llm = GradientBaseModelLLM(
        base_model_slug="llama2-7b-chat",
        max_tokens=500,
    )

    service_context = ServiceContext.from_defaults(node_parser=parser, embed_model=embed_model, llm=llm)
    set_global_service_context(service_context)
    index =  VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=False)
    index.storage_context.persist(index_path)
    print(f"{wikipage_requests} have been indexed.")
    return service_context

if __name__ == "__main__":
    wikitest = wikipedia.search("London", results=1)
    index_wikipedia_pages(wikitest)