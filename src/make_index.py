# Tool 1: Does a query based search for Wikipages
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser
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
from typing import Any, List
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
import wikipedia
import json
import os
import setup

setup.set_environment_variables()

# Define the path to the configurations file
config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configurations.json')
# Read and parse the JSON file
with open(config_file_path, 'r') as file:
    config = json.load(file)
# Extract configuration values
index_path = config['file_paths']['index_file']

# Functions for creating Wikipedia Index
def load_index(filepath: str):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    return load_index_from_storage(storage_context)

def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_wikidocs(wikipage_requests):
    print(f"Preparing to Download:{wikipage_requests}")
    documents = []
    for page_title in wikipage_requests:
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
    # nodes = parser.get_nodes_from_documents(documents)

    # Use the gradient embedding model
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

    service_context = ServiceContext.from_defaults(node_parser=parser, embed_model= embed_model, llm=llm)
    index =  VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=False)
    index.storage_context.persist(index_path)
    print(f"{wikipage_requests} have been indexed.")
    return "indexed"

def search_and_index_wikipedia(
        hops: list, lang: str = "en", results_limit: int = 2
    ):
    wikipedia.set_lang(lang)
    wikipage_requests = []
    for hop in hops:
        hop_pages = wikipedia.search(hop, results=results_limit)
        print(f"Searching Wikipedia for: {hop} - Found: {hop_pages}")
        wikipage_requests.extend(hop_pages)
    index_wikipedia_pages(wikipage_requests)

    return wikipage_requests


# def query_wiki_index(hops: List[str], index_path: str = index_path, n_results: int = 5): 
#     index = load_index(filepath=index_path)
#     query_engine = index.as_query_engine(
#         response_mode="compact", verbose=True, similarity_top_k=n_results
#     )
    
#     retrieved_context = {}
    
#     # Iterate over each hop in the multihop query
#     for hop in hops:
#         nodes = query_engine.query(hop).source_nodes
        
#         # Process each node found for the current hop
#         for node in nodes:
#             doc_id = node.node.id_
#             doc_text = node.node.text
#             doc_source = node.node.metadata.get('source_url', 'No source URL')  # Default value if source_url is not present.
            
#             # Append to the list of texts and sources for each doc_id
#             if doc_id not in retrieved_context:
#                 retrieved_context[doc_id] = {'texts': [doc_text], 'sources': [doc_source]}
#             else:
#                 retrieved_context[doc_id]['texts'].append(doc_text)
#                 retrieved_context[doc_id]['sources'].append(doc_source)

#     # Serialise the context for all hops into a JSON file
#     file_path = index_path + "retrieved_context.json"
#     with open(file_path, 'w') as f:
#         json.dump(retrieved_context, f)
    
#     return retrieved_context

if __name__ == "__main__":
    wikitest = wikipedia.search("London", results=1)
    index_wikipedia_pages(wikitest)