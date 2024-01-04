from llama_index.tools import QueryEngineTool, ToolMetadata
import chainlit as cl
from chainlit.input_widget import Select, TextInput
from llama_index import StorageContext, load_index_from_storage
from llama_index.agent import ReActAgent
from llama_index.chat_engine import ContextChatEngine
from llama_index.embeddings import GradientEmbedding
from llama_index.llms.gradient import _BaseGradientLLM
from llama_index.callbacks.base import CallbackManager
from typing import Any, List, Optional
from llama_index.bridge.pydantic import Field
import os
import json
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index import VectorStoreIndex, ServiceContext
# from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser import TokenTextSplitter
from llama_index.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from make_index import load_index, create_wikidocs
import nest_asyncio
nest_asyncio.apply()

# system_prompt = """[INST] <>
#     You are a helpful Wikipedia chat assistant. You goal is to 
#     answer questions based on the Wikipedia context you recieve.
#     You must always use the context to answer questions.
#     If a question does not make any sense, or is not factually coherent, explain 
#     why instead of answering something not correct. If you don't know the answer 
#     to a question, please don't share false information.
#       <>
#     """
#     # Throw together the query wrapper
# query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

class GradientBaseModelLLM(_BaseGradientLLM):
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

        )

        self._model = self._gradient.get_base_model(
            base_model_slug=base_model_slug,
        )

def index_wikipedia_pages(wikipage_requests):
    # Create a system prompt 
    system_prompt = """[INST] <>
    You are a helpful Wikipedia chat assistant. You goal is to 
    answer questions based on the Wikipedia context you recieve.
    You must always use the context to answer questions.
    If a question does not make any sense, or is not factually coherent, explain 
    why instead of answering something not correct. If you don't know the answer 
    to a question, please don't share false information.
      <>
    """
    # Throw together the query wrapper
    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")
     
    config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configurations.json')
    print("config_file_path", config_file_path)
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    index_path = config['file_paths']['index_file']
    print("index_path!!!", index_path)

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
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
    )   
    service_context = ServiceContext.from_defaults(node_parser=parser, embed_model=embed_model, llm=None)
    # set_global_service_context(service_context)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
    index.storage_context.persist(index_path)
    print(f"{wikipage_requests} have been indexed.")

    return index, service_context

def build_query_engine(index, service_context, n_results=5): 
    query_engine = index.as_query_engine(
        chat_mode='context', response_mode="compact", verbose=True, similarity_top_k=n_results, 
        service_context=service_context
    )
    return query_engine

def create_agent(wikipage_requests):
    # Create a system prompt 
    system_prompt = """[INST] <<SYS>>
    You are a helpful Wikipedia chat assistant. You goal is to 
    answer questions based on the Wikipedia context you recieve.
    You must always use the context to answer questions.
    If a question does not make any sense, or is not factually coherent, explain 
    why instead of answering something not correct. If you don't know the answer 
    to a question, please don't share false information. Answer only the {query_str}.
    do not generate additional questions and answers.
      <</SYS>>
    """
    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

    llm = GradientBaseModelLLM(
        base_model_slug="llama2-7b-chat",
        max_tokens=500,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        is_chat_model=True,
    )

    index, service_context = index_wikipedia_pages(wikipage_requests)
    retreiver = index.as_retriever(response_mode="compact", verbose=True, similarity_top_k=5, 
        service_context=service_context)

    agent = ContextChatEngine.from_defaults(
        llm=llm,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        verbose=True,
        retriever=retreiver,
        service_context=service_context
    )
    return agent

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
    # service_context = index_wikipedia_pages(wikipage_requests)
    # index = load_index(service_context)
    # print("on_settings_update", settings)
    # index, service_context = index_wikipedia_pages(wikipage_requests)
    query_engine = create_agent(wikipage_requests)
    # query_engine = build_query_engine(index=index, service_context=service_context, n_results=5)
    cl.user_session.set("query_engine", query_engine)
    await cl.Message(
        author="Wiki Agent", content=f"""Wikipage(s) "{wikipage_requests}" successfully indexed"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    response = await cl.make_async(query_engine.chat)(message.content)

    print(response)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()

    # await cl.Message(author="Agent", content=response).send()


# @cl.on_message
# async def main(message):
#     if agent:
#         response = await cl.make_async(agent.chat)(message)
#         # response = await cl.make_async(agent.chat(message))
#         await cl.Message(author="Agent", content=response).send()

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
