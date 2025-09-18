
import httpx
import requests
from fastapi import Body
from configs.model_config import DEFAULT_BIND_HOST
from server.knowledge_base.kb_doc_api import search_docs

async def twice_search(query:str = Body(..., description="用户输入", examples=["你好"]),
                        category: str = Body(..., description="问题类别", examples=["hunyin"])):

    docs = search_docs(query, "law_qa", 2, 0.50)
    if docs:
        async with httpx.AsyncClient() as client:
            response1 = await client.post("http://" + DEFAULT_BIND_HOST + ":7863/chat/qaandlaw",
                                          json={"query": query, "category": category})
            return response1.json()
    elif not docs:

        async with httpx.AsyncClient() as client:
            response2 = await client.post("http://" + DEFAULT_BIND_HOST + ":7863/chat/kb_chat_2search",
                                          json={"query": query, "category": category, "first_search_ans": ""})
            fs_ans = response2.json()["answer"]
            response3 = await client.post("http://" + DEFAULT_BIND_HOST + ":7863/chat/kb_chat_2search",
                                          json={"query": query, "category": category, "first_search_ans": fs_ans})
            return response3.json()
