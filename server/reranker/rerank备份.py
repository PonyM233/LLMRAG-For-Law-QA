from typing import Any, Optional, Sequence
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


def rerank_documents(query: str, documents: Sequence[Document],
                    top_n: int = 4, device: str = "cuda",
                    max_length: int = 1024, batch_size: int = 32,
                    num_workers: int = 0) -> Sequence[Document]:

    reranker_model_path = '/home/pc/PycharmProjects/chat0.2.4/bge-reranker-large'
    model = CrossEncoder(model_name=reranker_model_path, max_length=1024, device=device)
    print(documents)
    if not documents:
        return []

    doc_list = list(documents)
    _docs = [d.page_content for d in doc_list]

    sentence_pairs = [[query, _doc] for _doc in _docs]
    results = model.predict(sentences=sentence_pairs, batch_size=batch_size,
                            num_workers=num_workers, convert_to_tensor=True)

    top_k = min(top_n, len(results))
    values, indices = results.topk(top_k)

    final_results = []
    for value, index in zip(values, indices):
        doc = doc_list[index]
        doc.metadata["relevance_score"] = value
        final_results.append(doc)

    return final_results


if __name__ == "__main__":
    query = "your_query_here"
    documents = []  # Provide your list of documents
    reranked_docs = rerank_documents(query=query, documents=documents)
    print(reranked_docs)
'''
[DocumentWithScore(page_content='第一千零六十二条\u3000夫妻在婚姻关系存续期间所得的下列财产，为夫妻的共同财产，归夫妻共同所有：\n\n（一）工资、奖金、劳务报酬；\n\n（二）生产、经营、投资的收益；\n\n（三）知识产权的收益；\n\n（四）继承或者受赠的财产，但是本法第一千零六十三条第三项规定的除外；\n\n（五）其他应当归共同所有的财产。\n\n夫妻对共同财产，有平等的处理权。', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/中华人民共和国民法典.txt'}, score=0.31087732315063477), DocumentWithScore(page_content='第一千零六十三条\u3000下列财产为夫妻一方的个人财产：\n\n（一）一方的婚前财产；\n\n（二）一方因受到人身损害获得的赔偿或者补偿；\n\n（三）遗嘱或者赠与合同中确定只归一方的财产；\n\n（四）一方专用的生活用品；\n\n（五）其他应当归一方的财产。', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/中华人民共和国民法典.txt'}, score=0.32154780626296997), DocumentWithScore(page_content='第一千零六十五条\u3000男女双方可以约定婚姻关系存续期间所得的财产以及婚前财产归各自所有、共同所有或者部分各自所有、部分共同所有。约定应当采用书面形式。没有约定或者约定不明确的，适用本法第一千零六十二条、第一千零六十三条的规定。\n\n夫妻对婚姻关系存续期间所得的财产以及婚前财产的约定，对双方具有法律约束力。\n\n夫妻对婚姻关系存续期间所得的财产约定归各自所有，夫或者妻一方对外所负的债务，相对人知道该约定的，以夫或者妻一方的个人财产清偿。', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/中华人民共和国民法典.txt'}, score=0.3441479802131653), DocumentWithScore(page_content='第七十二条\u3000夫妻双方分割共同财产中的股票、债券、投资基金份额等有价证券以及未上市股份有限公司股份时，协商不成或者按市价分配有困难的，人民法院可以根据数量按比例分配。', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释(一).txt'}, score=0.347380667924881), DocumentWithScore(page_content='第二十六条\u3000夫妻一方个人财产在婚后产生的收益，除孳息和自然增值外，应认定为夫妻共同财产。', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释(一).txt'}, score=0.35066789388656616), DocumentWithScore(page_content='第二十五条\u3000婚姻关系存续期间，下列财产属于民法典第一千零六十二条规定的“其他应当归共同所有的财产”：\n\n（一）一方以个人财产投资取得的收益；\n\n（二）男女双方实际取得或者应当取得的住房补贴、住房公积金；\n\n（三）男女双方实际取得或者应当取得的基本养老金、破产安置补偿费。', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释(一).txt'}, score=0.35421687364578247), DocumentWithScore(page_content='第二十二条\u3000被确认无效或者被撤销的婚姻，当事人同居期间所得的财产，除有证据证明为当事人一方所有的以外，按共同共有处理。\n\n三、夫妻关系', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释(一).txt'}, score=0.3553187847137451), DocumentWithScore(page_content='第一千零九十二条\u3000夫妻一方隐藏、转移、变卖、毁损、挥霍夫妻共同财产，或者伪造夫妻共同债务企图侵占另一方财产的，在离婚分割夫妻共同财产时，对该方可以少分或者不分。离婚后，另一方发现有上述行为的，可以向人民法院提起诉讼，请求再次分割夫妻共同财产。\n\n第五章\u3000收养\n\n第一节\u3000收养关系的成立', metadata={'source': '/home/pc/PycharmProjects/chat0.2.4/knowledge_base/law/content/中华人民共和国民法典.txt'}, score=0.35532286763191223)]

'''