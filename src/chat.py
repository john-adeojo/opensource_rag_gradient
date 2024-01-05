import chainlit as cl
import wikipedia
from chainlit.input_widget import TextInput
from llama_index.embeddings import GradientEmbedding
from llama_index.readers.schema.base import Document
from llama_index.llms.gradient import _BaseGradientLLM
from llama_index.callbacks.base import CallbackManager
from typing import Optional
from llama_index.prompts import PromptTemplate
from llama_index.bridge.pydantic import Field
import os
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.node_parser import TokenTextSplitter
import setup

setup.set_environment_variables()

class CustomGradientBaseModelLLM(_BaseGradientLLM):
    base_model_slug: str = Field(
        description="The slug of the base model to use.",
    )

    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        base_model_slug: str,
        host: Optional[str] = None,
        max_tokens: Optional[int] = None,
        workspace_id: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_chat_model: bool = False,
        system_prompt: Optional[str] = None,
        query_wrapper_prompt: Optional[str] = None,
        # messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        # completion_to_prompt: Optional[Callable[[str], str]] = None,

    ) -> None:
        super().__init__(
            access_token=access_token,
            base_model_slug=base_model_slug,
            host=host,
            max_tokens=max_tokens,
            workspace_id=workspace_id,
            callback_manager=callback_manager,
            is_chat_model=is_chat_model,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            # messages_to_prompt=messages_to_prompt,
            # completion_to_prompt=completion_to_prompt,
        )

        self._model = self._gradient.get_base_model(
            base_model_slug=base_model_slug,
        )

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

    # Use the open source LLMs hosted by Gradient
    query_wrapper_prompt = PromptTemplate("[INST] {response} [/INST]")
    llm = CustomGradientBaseModelLLM(
        base_model_slug="llama2-7b-chat",
        max_tokens=250,
        is_chat_model=True,
        query_wrapper_prompt=query_wrapper_prompt
    )  

    print(f"Preparing to index Wikipages: {wikipage_requests}")
    documents = create_wikidocs(wikipage_requests)
    parser = TokenTextSplitter(chunk_size=150, chunk_overlap=45)
    
    embed_model = GradientEmbedding(
    gradient_access_token=os.environ['GRADIENT_ACCESS_TOKEN'],
    gradient_workspace_id=os.environ['GRADIENT_WORKSPACE_ID'],
    gradient_model_slug="bge-large",
    )
 
    service_context = ServiceContext.from_defaults(node_parser=parser, embed_model=embed_model, llm=llm)
    # set_global_service_context(service_context)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
    # index.storage_context.persist(index_path)
    print(f"{wikipage_requests} have been indexed.")

    return index, service_context

def build_query_engine(wikipage_requests, n_results=5): 
    index, service_context = index_wikipedia_pages(wikipage_requests)
    query_engine = index.as_query_engine(
        chat_mode='context', 
        response_mode="compact", 
        verbose=True, 
        similarity_top_k=n_results, 
        service_context=service_context
    )
    return query_engine

@cl.on_chat_start
async def on_chat_start():
    # Settings
    settings = await cl.ChatSettings(
        [
            TextInput(id="WikiPageRequest", label="Request Wikipage(s)"),
        ]
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    global agent
    global index
    wikipage_requests = settings["WikiPageRequest"]
    query_engine = build_query_engine(wikipage_requests, n_results=5)
    cl.user_session.set("query_engine", query_engine)
    await cl.Message(
        author="Wiki Agent", content=f"""Wikipage(s) "{wikipage_requests}" successfully indexed"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    response = await cl.make_async(query_engine.query)(message.content)
    print(response)
    response_message = cl.Message(content=response)
    print("response message", response_message)
    await cl.Message(author="Agent", content=response_message.content).send()

# if __name__ == "__main__":
    # index, service_context = index_wikipedia_pages(["2023 United States banking crisis"])
    # query_engine = query_wiki_index(index=index, n_results=5, service_context=service_context)
    # response = query_engine.chat("Which bank was first to default?")
    # print(response)
    # response = query_engine.query("Which bank was first to default?")
    # print(response)
    # agent = create_agent(["2023 United States banking crisis"])
    # response = agent.chat("Which bank was first to default?")
    # print(response)
