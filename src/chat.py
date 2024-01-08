import chainlit as cl
import wikipedia
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
)
from chainlit.input_widget import TextInput, Select
from llama_index.embeddings import GradientEmbedding
from llama_index.readers.schema.base import Document
from llama_index.llms.gradient import _BaseGradientLLM
from llama_index.callbacks.base import CallbackManager
from typing import Optional, Any, Callable, Sequence
from llama_index.prompts import PromptTemplate
from llama_index.bridge.pydantic import Field
import os
from typing_extensions import override
from llama_index.llms.types import CompletionResponse
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.node_parser import TokenTextSplitter
# from llama_index.llms.gradient import GradientBaseModelLLM, GradientModelAdapterLLM
import setup

# Parameters
chunk_size = 150
chunk_overlap = 30

setup.set_environment_variables()
global model_adapter_id
model_adapter_id = "d74be430-903b-4f18-9162-b06f81aeb4d5_model_adapter"

qa_prompt = PromptTemplate(
    "<s>### Instruction:\n"
    "Using the context, respond like Yoda to: {query_str}\n\n"
    "### Input:\n"
    "{context_str}\n\n"
    "### Response:\n"
)

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
    ) -> None:
        super().__init__(
            access_token=access_token,
            base_model_slug=base_model_slug,
            host=host,
            max_tokens=max_tokens,
            workspace_id=workspace_id,
            callback_manager=callback_manager,
            is_chat_model=is_chat_model,
        )

        self._model = self._gradient.get_base_model(
            base_model_slug=base_model_slug,
        )
 
    @override
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print(f"Custom Prompt: {prompt}")  # Custom print statement
        return super().complete(prompt, temperature=1, **kwargs)

    @override
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print(f"Custom Prompt: {prompt}")  # Custom print statement
        return await super().acomplete(prompt, temperature=1, **kwargs)
    

class CustomGradientModelAdapterLLM(_BaseGradientLLM):
    model_adapter_id: str = Field(
        description="The id of the model adapter to use.",
    )

    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        host: Optional[str] = None,
        max_tokens: Optional[int] = None,
        model_adapter_id: str,
        workspace_id: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_chat_model: bool = False,
    ) -> None:
        super().__init__(
            access_token=access_token,
            host=host,
            max_tokens=max_tokens,
            model_adapter_id=model_adapter_id,
            workspace_id=workspace_id,
            callback_manager=callback_manager,
            is_chat_model=is_chat_model,
        )
        self._model = self._gradient.get_model_adapter(
            model_adapter_id=model_adapter_id,
        )

    @override
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print(f"Custom Prompt: {prompt}")  # Custom print statement
        return super().complete(prompt, temperature=1, **kwargs)

    @override
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print(f"Custom Prompt: {prompt}")  # Custom print statement
        return await super().acomplete(prompt, temperature=1, **kwargs)

class RAGStringQueryEngine_Adpater(CustomQueryEngine):
    """RAG String Query Engine."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: CustomGradientModelAdapterLLM
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        kwargs = {"temperature":1}
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str, **kwargs)
        )

        return str(response)
    
class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: CustomGradientBaseModelLLM
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        kwargs = {"temperature":1}
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str, **kwargs)
        )

        return str(response)


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

def index_wikipedia_pages(wikipage_requests, settings):

    # Use the open source LLMs hosted by Gradient
    if settings['MODEL'] == "llama2-7b-chat":
        llm = CustomGradientBaseModelLLM(
            base_model_slug="llama2-7b-chat",
            max_tokens=500,
            is_chat_model=True,
        )

    elif settings['MODEL'] == "nous-hermes2-yoda":
        llm = CustomGradientModelAdapterLLM(
            model_adapter_id=model_adapter_id,
            max_tokens=500,
            is_chat_model=True,

        )
    elif settings['MODEL'] == "nous-hermes2":
        llm = CustomGradientBaseModelLLM(
            base_model_slug="nous-hermes2",
            max_tokens=500,
            is_chat_model=True,
        )

    print(f"Preparing to index Wikipages: {wikipage_requests}")
    documents = create_wikidocs(wikipage_requests)
    parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    embed_model = GradientEmbedding(
    gradient_access_token=os.environ['GRADIENT_ACCESS_TOKEN'],
    gradient_workspace_id=os.environ['GRADIENT_WORKSPACE_ID'],
    gradient_model_slug="bge-large",
    )
 
    service_context = ServiceContext.from_defaults(node_parser=parser, embed_model=embed_model, llm=llm)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
    print(f"{wikipage_requests} have been indexed.")

    return index, service_context, llm

def build_query_engine(wikipage_requests, n_results, settings): 
    index, service_context, llm = index_wikipedia_pages(wikipage_requests, settings)
    synthesizer = get_response_synthesizer(response_mode="compact", service_context=service_context)

    retriever = index.as_retriever(
        response_mode="compact", 
        verbose=True, 
        similarity_top_k=n_results,
    )

    if settings['MODEL'] == "nous-hermes2-yoda":
        query_engine = RAGStringQueryEngine_Adpater(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
            service_context=service_context
        )
    else:
        query_engine = RAGStringQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
            service_context=service_context
        )

    return query_engine, llm

# def build_query_engine(wikipage_requests, n_results, settings): 
#     index, service_context, llm = index_wikipedia_pages(wikipage_requests, settings)
#     query_engine = index.as_query_engine(
#         chat_mode='context', 
#         response_mode="compact", 
#         verbose=True, 
#         similarity_top_k=n_results, 
#         service_context=service_context
#     )
#     return query_engine, llm

@cl.on_chat_start
async def on_chat_start():
    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="MODEL",
                label="Gradient Model",
                values=["nous-hermes2-yoda", "llama2-7b-chat", "nous-hermes2"],
                initial_index=0,
            ),
            TextInput(id="WikiPageRequest", label="Request Wikipage(s)"),
        ]
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    global agent
    global index
    wikipage_requests = settings["WikiPageRequest"]
    query_engine, llm = build_query_engine(wikipage_requests=wikipage_requests, n_results=5, settings=settings)
    cl.user_session.set("query_engine", query_engine)
    cl.user_session.set("llm", llm)
    await cl.Message(
        author="Wiki Agent", content=f"""Wikipage(s) "{wikipage_requests}" successfully indexed
        You are using the {settings['MODEL']} model.
        The model id is {llm._model.id}
        """
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    llm = cl.user_session.get("llm")
    print("TEMPLATE", llm)
    response = await cl.make_async(query_engine.query)(message.content)
    print(response)
    response_message = cl.Message(content=response)
    print("response message", response_message)
    await cl.Message(author="Agent", content=response_message.content).send()
