"""Tool for load HTML files and QA."""
from typing import Any, List, Optional, Type

from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.requests import TextRequestsWrapper
from langchain.tools.base import BaseTool
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.tools.base import ToolException


# def _clean_url(url: str) -> str:
#     """Strips quotes from the url."""
#     return url.strip("\"'")


class WebQASchema(BaseModel):
    question: str = Field(description="should be a question on response content")
    urls: List[str] = Field(description="should be a list of strings")


class WebQA(BaseTool):
    """Tool for load HTML files and QA."""

    name = "webpage_qa"
    description = "Use this when you need to answer questions about specific webpages"
    args_schema: Type[WebQASchema] = WebQASchema
    llm: Any = Field()
    embeddings: Any = Field()

    def _run(
        self,
        question: str,
        urls: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        print("WebQA")
        try:
            loader = WebBaseLoader(web_path=urls)
            index = VectorstoreIndexCreator(embedding=self.embeddings).from_loaders(
                [loader]
            )
            answer = index.query(question=question, llm=self.llm)
            return answer
        except IOError as e:
            raise ToolException() from e

    async def _arun(
        self,
        question: str,
        urls: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        search_query = f"site:wikipedia.org {question}"  # Modify search query as needed
        search_results = await self.requests_tool.arun(url=f"https://www.google.com/search?q={search_query}")
        # Process search_results and extract relevant information for answering the question
        # Perform question answering using the extracted information
        answer = "Sample answer"  # Replace with actual answer
        return answer

        return await self.requests_wrapper.aget(_clean_url(url))
