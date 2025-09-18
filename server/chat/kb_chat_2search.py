from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                                  TEMPERATURE)
from server.chat.utils import wrap_done
from server.reranker.reranker import rerank_documents
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
log_file = "/home/pc/PycharmProjects/chat0.2.4/log/11_21.txt"
import datetime

'''
def extract_query(sentence):
    try:
        # 尝试解析sentence为JSON格式
        sentence_dict = json.loads(sentence)
        if "question" in sentence_dict:
            return sentence_dict["question"]
    except json.JSONDecodeError:
        # 如果无法解析为JSON，直接返回原句
        return sentence

    # 如果不是JSON格式，直接返回原句
    return sentence
'''
def process_text(text):
    # 检测第一句中是否包含特定字符串
    if "法律顾问" in text.split('。')[0]:
        # 删除第一句
        text = '。'.join(text.split('。')[1:])
    # 检测删除第一句后的文本是否以特定关键词开头
    if text.startswith("但是，") or text.startswith("然而，"):
        # 删除开头的关键词
        text = text[len("但是，"):] if text.startswith("但是，") else text[len("然而，"):]
    return text
async def kb_chat_2search(query: str = Body(..., description="用户输入", examples=["你好"]),
                            category: str = Body(..., description="问题类别", examples=["hunyin"]),
                            first_search_ans: str = Body(..., description="第一次搜索的回答", examples=["你好"]),
                            knowledge_base_name: str = Body("law", description="知识库名称", examples=["samples"]),
                            top_k: int = Body(8, description="匹配向量数"),
                            score_threshold: float = Body(0.53, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=1),#0.53
                            history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                            stream: bool = Body(False, description="流式输出"),
                            model_name: str = Body("Baichuan2-13B", description="LLM 模型名称。"),
                            temperature: float = Body(0.05, description="LLM 采样温度", gt=0.0, le=1.0),
                            local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                            request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODEL,
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()

        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
            openai_api_key=llm_model_dict[model_name]["api_key"],
            openai_api_base=llm_model_dict[model_name]["api_base_url"],
            model_name=model_name,
            temperature=temperature,
            openai_proxy=llm_model_dict[model_name].get("openai_proxy")
        )
        query1 = query

        score = 0.53
        if "三倍" in query:
            score = 0.9

        knowledge_base_name1 = knowledge_base_name + "_anli" +"_" + category
        knowledge_base_name2 = knowledge_base_name + "_qa"
        knowledge_base_name_anliyuanwenj = knowledge_base_name + "_anliyuanwenj"
        if not first_search_ans:
            docs = search_docs(query, knowledge_base_name, top_k, score)
            docs1 = search_docs(query, knowledge_base_name1, 2, 0.6)
            docs2 = search_docs(query, knowledge_base_name2, 2, 0.5)
        elif first_search_ans:
            docs = search_docs(query, knowledge_base_name, top_k, score)
            docs1 = search_docs(first_search_ans, knowledge_base_name1, 2, 0.6)
            docs2 = search_docs(first_search_ans, knowledge_base_name2, 2, 0.5)
        #context = "\n".join([doc.page_content for doc in docs])
        docs = rerank_documents(top_n=4, documents=docs, query=query)

        context = ""
        # 迭代处理每个文档并将文件名信息和文档内容添加到context
        for inum, doc in enumerate(docs2):
            full_filename = os.path.split(doc.metadata["source"])[-1]  # 生成文件名信息和文档内容的文本，并将其添加到context中
            filename, file_extension = os.path.splitext(full_filename)
            doc_info = f" {filename}\n{doc.page_content}\n\n"
            context += doc_info

        for inum, doc in enumerate(docs):
            full_filename = os.path.split(doc.metadata["source"])[-1]  # 生成文件名信息和文档内容的文本，并将其添加到context中
            filename, file_extension = os.path.splitext(full_filename)
            doc_info = f" {filename}\n{doc.page_content}\n\n"
            context += doc_info

        for inum, doc in enumerate(docs1):
            full_filename = os.path.split(doc.metadata["source"])[-1]  # 生成文件名信息和文档内容的文本，并将其添加到context中
            filename, file_extension = os.path.splitext(full_filename)
            doc_info = f" {filename}\n{doc.page_content}\n\n"
            context += doc_info

        input_msg = History(role="user", content=PROMPT_TEMPLATE).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)
        response_paragraph = ""

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        ########################################
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
                # text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
                text = f""" [{inum + 1}] 《{filename}》 {extracted_content}{'条'}\n"""
                source_documentslawn.append(text)

        source_documentslawc = []
        for inum, doc in enumerate(docs):
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                text = f"""{doc.page_content}\n\n"""
                # text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
                source_documentslawc.append(text)
        ########################################
        ########################################
        source_documentslawn2 = []
        for inum, doc in enumerate(docs):
            if inum == 1:
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                first_space_index = doc.page_content.find('条')
                extracted_content = doc.page_content[:first_space_index]
                if local_doc_url:
                    url = "file://" + doc.metadata["source"]
                else:
                    parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                    url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                # text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
                text = f""" [{inum + 1}] 《{filename}》 {extracted_content}{'条'}\n\n"""
                source_documentslawn2.append(text)

        source_documentslawc2 = []
        for inum, doc in enumerate(docs):
            if inum == 1:
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                text = f"""{doc.page_content}\n\n"""
                # text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
                source_documentslawc2.append(text)
        ########################################
        ########################################
        source_documentslawn3 = []
        for inum, doc in enumerate(docs):
            if inum == 2:
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                first_space_index = doc.page_content.find('条')
                extracted_content = doc.page_content[:first_space_index]
                if local_doc_url:
                    url = "file://" + doc.metadata["source"]
                else:
                    parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                    url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                # text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
                text = f""" [{inum + 1}] 《{filename}》 {extracted_content}{'条'}\n\n"""
                source_documentslawn3.append(text)

        source_documentslawc3 = []
        for inum, doc in enumerate(docs):
            if inum == 2:
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                text = f"""{doc.page_content}\n\n"""
                # text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
                source_documentslawc3.append(text)
        ########################################
        ########################################
        source_documentslawn4 = []
        for inum, doc in enumerate(docs):
            if inum == 3:
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                first_space_index = doc.page_content.find('条')
                extracted_content = doc.page_content[:first_space_index]
                if local_doc_url:
                    url = "file://" + doc.metadata["source"]
                else:
                    parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                    url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                # text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
                text = f""" [{inum + 1}] 《{filename}》 {extracted_content}{'条'}\n\n"""
                source_documentslawn4.append(text)

        source_documentslawc4 = []
        for inum, doc in enumerate(docs):
            if inum == 3:
                filename_with_extension = os.path.split(doc.metadata["source"])[-1]
                filename, _ = os.path.splitext(filename_with_extension)
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
                text = f"""{doc.page_content}\n\n"""
                # text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
                source_documentslawc4.append(text)
        ########################################

#        source_documents = []#法律
#        for inum, doc in enumerate(docs):
#            filename_with_extension = os.path.split(doc.metadata["source"])[-1]
#            filename, _ = os.path.splitext(filename_with_extension)
#            first_space_index = doc.page_content.find('条')
#            extracted_content = doc.page_content[:first_space_index]
#            if local_doc_url:
#                url = "file://" + doc.metadata["source"]
#            else:
#                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
#                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
#            #text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
#            text = f""" [{inum + 1}] 《{filename}》 {extracted_content}{'条'}\n\n{doc.page_content}\n\n"""
#            source_documents.append(text)

        source_documents1 = []#案例名
        for inum, doc in enumerate(docs1):
            filename_with_extension1 = os.path.split(doc.metadata["source"])[-1]
            filename1, _ = os.path.splitext(filename_with_extension1)
            first_newline_index = doc.page_content.find(':')
            extracted_content_anliname = doc.page_content[:first_newline_index]
            extracted_content_anlinamewithextension = extracted_content_anliname + ".docx"
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name_anliyuanwenj, "file_name": extracted_content_anlinamewithextension})
                url = f"https://52258l5q66.zicp.fun/knowledge_base/download_doc?" + parameters
            #text = f"""出处 [{inum + 1}] [{filename1}]({url}) \n\n{doc.page_content}\n\n"""

            text = f""" [{inum + 1}] [{extracted_content_anliname}] \n{url}\n"""
            #text = f"""出处 [{inum + 1 + top_k}] [{filename1}] \n\n{doc.page_content}\n\n"""
            source_documents1.append(text)

        source_documents2 = []#问答
        for inum, doc in enumerate(docs2):
            filename_with_extension2 = os.path.split(doc.metadata["source"])[-1]
            filename2, _ = os.path.splitext(filename_with_extension2)
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name2, "file_name": filename2})
                url = f"https://52258l5q66.zicp.fun/knowledge_base/download_doc?" + parameters
            # text = f"""出处 [{inum + 1}] [{filename1}]({url}) \n\n{doc.page_content}\n\n"""
            text = f""" [{inum + 1}] [{filename2}] \n\n{doc.page_content}\n\n"""
            # text = f"""出处 [{inum + 1 + top_k}] [{filename1}] \n\n{doc.page_content}\n\n"""
            source_documents2.append(text)

        source_documents3 = []  # 案例内容
        for inum, doc in enumerate(docs1):
            filename_with_extension = os.path.split(doc.metadata["source"])[-1]
            filename, _ = os.path.splitext(filename_with_extension)
            text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
            source_documents3.append(text)


        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
                response_paragraph += token
            yield json.dumps({"lawname": source_documentslawn}, ensure_ascii=False)
            yield json.dumps({"lawcont": source_documentslawc}, ensure_ascii=False)
            yield json.dumps({"docs1": source_documents1}, ensure_ascii=False)
            yield json.dumps({"docs2": source_documents2}, ensure_ascii=False)
            yield json.dumps({"docs3": source_documents3}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
                response_paragraph += token

            if response_paragraph[:17] == "很抱歉，我们没能很好地理解您的问题":
                response_paragraph = "很抱歉，我们没能很好地理解您的问题，您可以试着换一个问法。"
            if answer[:17] == "很抱歉，我们没能很好地理解您的问题":
                answer = "很抱歉，我们没能很好地理解您的问题，您可以试着换一个问法。"
            #if "您好，我们将结合人工智能技术尝试回答您的问题。" in answer:
            #    answer = answer.replace("您好，我们将结合人工智能技术尝试回答您的问题。", "您好，我们的专业知识库暂未收录您的问题，但我们会结合互联网并通过人工智能技术尝试回答您的问题：\n\n")

            if "当地法律法规的规定" in answer:
                answer = answer.replace("当地法律法规的规定", "相关法律规定")
            if "当地的法律法规的规定" in answer:
                answer = answer.replace("当地的法律法规的规定", "相关法律规定")
            if "当地的法律法规" in answer:
                answer = answer.replace("当地的法律法规", "相关法律规定")
            if "根据已知信息，" in answer:
                answer = answer.replace("根据已知信息，", "")
            if "根据您提供的信息，"in answer:
                answer = answer.replace("根据您提供的信息，", "")
            if "根据提供的已知信息，"in answer:
                answer = answer.replace("根据提供的已知信息，", "")
            if "大语言模型"in answer:
                answer = answer.replace("大语言模型", "智能法律顾问")
            if "在中国，"in answer:
                answer = answer.replace("在中国，", "")
            if "请记住，我只是一个AI助手" in answer:
                answer = answer.replace("请记住，我只是一个AI助手", "我不是专业的律师")
            if "您所在国家/地区" in answer:
                answer = answer.replace("您所在国家/地区", "相关")
            def check_keywords(answer):
                keywords = ["刑法", "酒驾","量刑","醉驾"]

                for keyword in keywords:
                    if keyword in answer:
                        return "很抱歉，我们主要专注于解决民事类法律问题，对于您的问题，我们可能无法提供专业和准确的建议，感谢您的理和支持"

                return answer

            answer = process_text(answer)
            #answer = check_keywords(answer)

            if "当地法律法规的规定" in response_paragraph:
                response_paragraph = response_paragraph.replace("当地法律法规的规定", "相关法律规定")
            if "当地的法律法规的规定" in response_paragraph:
                response_paragraph = response_paragraph.replace("当地的法律法规的规定", "相关法律规定")
            if "当地的法律法规" in response_paragraph:
                response_paragraph = response_paragraph.replace("当地的法律法规", "相关法律规定")
            if "根据已知信息，" in response_paragraph:
                response_paragraph = response_paragraph.replace("根据已知信息，", "")
            if "根据您提供的信息，" in response_paragraph:
                response_paragraph = response_paragraph.replace("根据您提供的信息，", "")
            if "根据提供的已知信息，"in response_paragraph:
                response_paragraph = response_paragraph.replace("根据提供的已知信息，", "")

            #response_paragraph = check_keywords(response_paragraph)

            if source_documentslawn ==[] and source_documents1 ==[] and source_documents2 ==[] :
                answer = "很抱歉，我们没能很好地理解您的意图或问题，请您提出一个具体的法律问题或者试着换一个问法。"
            #    response_paragraph = "很抱歉，我们没能很好地理解您的意图或问题，请您提出一个具体的法律问题或者试着换一个问法。"

            yield json.dumps({"answer": answer,
                              "lawname": source_documentslawn,
                              "lawcont": source_documentslawc,
                              "docs1": source_documents1,
                              "docs2": source_documents2,
                              "docs3": source_documents3,
                              },
                             ensure_ascii=False)
        current_datetime = datetime.datetime.now()
        with open(log_file, "a") as f:
            #f.write(f"提示词 : {PROMPT_TEMPLATE}\n")
            #f.write(f"匹配数 : {top_k} ，阈值 : {score_threshold}\n")

            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"提问时间: {formatted_datetime}\n")
            #data = json.loads(query)
            # 提取"question"键对应的值
            #query1 = data.get("question", "")
            #query2 = extract_query(query1)
            formatted_question = f"问题：{query1}"
            f.write(formatted_question + "\n")
            f.write(f"回复: {response_paragraph}\n\n\n")


        await task

    return StreamingResponse(knowledge_base_chat_iterator(query, kb, top_k, history, model_name),
                             media_type="text/event-stream")
