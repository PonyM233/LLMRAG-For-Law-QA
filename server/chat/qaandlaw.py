from fastapi import Body, Request
from fastapi.responses import StreamingResponse

from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.chat.qa_search_answer import search_answers_in_directory
from configs.model_config import qa_base_path
from server.reranker.reranker import rerank_documents


async def qaandlaw(query: str = Body(..., description="用户输入", examples=["你好"]),
                            category: str = Body(..., description="问题类别", examples=["hunyin"]),
                            knowledge_base_name: str = Body("law", description="知识库名称", examples=["samples"]),
                            top_k: int = Body(8, description="匹配向量数"),
                            score_threshold: float = Body(0.45, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=1),
                            local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                            request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)

    async def qaandlaw_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()

        knowledge_base_name1 = knowledge_base_name + "_anli" + "_" + category
        knowledge_base_name_anliyuanwenj = knowledge_base_name + "_anliyuanwenj"

        docs = search_docs(query.strip('？'), knowledge_base_name, top_k, score_threshold)
        docs2 = search_docs(query.strip('？'), knowledge_base_name1, 2, 0.6)
        #results = "您好，根据您的描述，我的理解是:\n\n" + "".join(search_answers_in_directory(query, qa_base_path))
        results = "".join(search_answers_in_directory(query, qa_base_path))

        docs = rerank_documents(top_n=4, documents=docs,query=query)

        source_documentslawn = []
        for inum, doc in enumerate(docs):
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                first_space_index = doc.page_content.find('条')
                extracted_content = doc.page_content[:first_space_index]
                if local_doc_url:
                    url = "file://" + doc.metadata["source"]
                else:
                    parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                    url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                text = f""" [{inum + 1}] 《{filename}》 {extracted_content}{'条'}\n"""
                source_documentslawn.append(text)

        source_documentslawc = []
        for inum, doc in enumerate(docs):
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                text = f"""{doc.page_content}\n\n"""
                source_documentslawc.append(text)


        source_documents2 = []  # 案例
        for inum, doc in enumerate(docs2):
            filename_with_extension1 = os.path.split(doc.metadata["source"])[-1]
            filename1, _ = os.path.splitext(filename_with_extension1)
            first_newline_index = doc.page_content.find(':')
            extracted_content_anliname = doc.page_content[:first_newline_index]
            extracted_content_anlinamewithextension = extracted_content_anliname + ".docx"
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name_anliyuanwenj,
                                        "file_name": extracted_content_anlinamewithextension})
                url = f"https://52258l5q66.zicp.fun/knowledge_base/download_doc?" + parameters

            text = f""" [{inum + 1}] [{extracted_content_anliname}] \n{url}\n"""
            source_documents2.append(text)

        result_dict = {

            "answer":  results,
            "lawname": source_documentslawn,
            "lawcont": source_documentslawc,
            "docs1": source_documents2
        }

            # Serialize the result_dict only once and yield it
        result_json = json.dumps(result_dict, ensure_ascii=False)
        yield result_json

    return StreamingResponse(qaandlaw_iterator(query, kb, top_k),
                                 media_type="text/event-stream")
