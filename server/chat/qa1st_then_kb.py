#先把问题与问题库进行匹配，如果能匹配到，那么将匹配到的作为新的query，然后调用qa生成答案
#                     如果匹配不到，将问题送给kbchat
import json

import httpx
import requests
from fastapi import Body
from configs.model_config import DEFAULT_BIND_HOST
from server.knowledge_base.kb_doc_api import search_docs
from server.chat.qa_search_answer import search_answers_in_directory
from configs.model_config import qa_base_path
async def qa1st_then_kb(query:str = Body(..., description="用户输入", examples=["你好"]),
                        category: str = Body(..., description="问题类别", examples=["hunyin"])):

    docs = search_docs(query, "law_allquestion", 2, 0.15)
    #docs1 = search_docs(query, "law_qa", 1, 0.50)

    docs2 = search_docs(query, "law_allquestion", 1, 0.25)

    if docs:
        query1 = docs[0].page_content
        async with httpx.AsyncClient() as client:
            response1 = await client.post("http://" + DEFAULT_BIND_HOST + ":7863/chat/qaandlaw",
                                          json={"query": query1, "category": category})
            return response1.json()

    elif docs == [] and docs2:
        response3 = []
        answers = []

        for i in range(len(docs2)):
            if docs2[i].page_content:
                answers.append(docs2[i].page_content)
                results = search_answers_in_directory(docs2[i].page_content, qa_base_path)
        if answers:
            answer = "您好，根据您的描述，我觉得你可能想知道：\n\n" + '\n'.join(answers) + '\n\n' + ''.join(results)
            #response3.append({"answer": answer})

        return {"answer":answer}



'''
result=qaandlaw(query="夫妻共同财产有哪些",category="hunyin",knowledge_base_name="law")
loop = asyncio.get_event_loop()
loop.run_until_complete(result)
result_json = result.body.decode('utf-8')
aaa=result_json
async def qa1st_then_kb(query: str = Body(..., description="用户输入", examples=["你好"]),
                            category: str = Body(..., description="问题类别", examples=["hunyin"]),
                            knowledge_base_name: str = Body("law", description="知识库名称", examples=["samples"]),
                            top_k: int = Body(1, description="匹配向量数"),
                            score_threshold: float = Body(0.45, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=1),

                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    async def qa1st_then_kb_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()

        docs = search_docs(query, "law_allquestion", 1, 0.5)
        if docs[0].page_content != "":
            query = docs[0].page_content
            result=qaandlaw(query=query,category=category)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(result)
            result_json = result.body.decode('utf-8')
            yield result_json
        elif docs[0].page_content == "":
            result_json = json.dumps(knowledge_base_chat(query=query,category=category), ensure_ascii=False)
            yield result_json

    return StreamingResponse(qa1st_then_kb_iterator(query, kb, top_k),
                                 media_type="text/event-stream") 
                                '''