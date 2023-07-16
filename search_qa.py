"""Tool that calls SerpAPI and QA.
"""
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, Tool, tool

from typing import Optional, Type, Any

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from serp_loader import SerpAPILoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import logging
import sys


class SearchSchema(BaseModel):
    question: str = Field(description="should be a question on Google search results")
    query: str = Field(
        description="should be a Google search query. You can use anything that you would use in a regular Google search."
    )


class SearchQA(BaseTool):
    name = "search_qa"
    description = "Useful for when you need to answer questions about current events"
    args_schema: Type[SearchSchema] = SearchSchema
    llm: Any = Field()
    serp: Any = Field()
    embeddings: Any = Field()

    def _run(
        self,
        question: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        print("SearchQA Tool")
        print(f"Question: {question}")
        print(f"Query: {query}")
        loader = SerpAPILoader(query, self.serp)
        # chain = load_qa_chain(llm)
        # chain.run(input_documents=loader.load(), question="LangChain 공식 문서 주소 검색해봐")
        index = VectorstoreIndexCreator(embedding=self.embeddings).from_loaders(
            [loader]
        )
        # query = "LangChain 공식 문서 link는?"
        # index.query_with_sources(question=query, llm=llm)

        res = index.query(question=question, llm=self.llm)
        return res

    async def _arun(
        self,
        question: str,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("search_qa does not support async")
