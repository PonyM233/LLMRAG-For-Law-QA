# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class anlitextsplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = ["号:"]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        current_chunk = ''
        _separator = separators[0] if self._is_separator_regex else re.escape(separators[0])
        for line in text.splitlines():
            if re.search(_separator, line):
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ''
            current_chunk += line + '\n'

        if current_chunk:
            final_chunks.append(current_chunk.strip())

        return final_chunks

if __name__ == "__main__":
    text_splitter = anlitextsplitter(
        separators=["？"],
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=200,
        chunk_overlap=0
    )

    text = """（2022）浙0109民初6443号:
（类案背景与基础法理）
在现实生活中，个别用人单位因为没有缴纳工伤保险，在发生重大工伤事故时，往往选择以恶意关闭企业的方式逃避工伤保险待遇费用的支付责任，但这并不能逃避赔偿责任。
（法律条款适用）
结合具体法律规定来看，《工伤保险条例》第三十七条规定，职工因工致残被鉴定为七级至十级伤残的，享受以下待遇：(一)从工伤保险基金按伤残等级支付一次性伤残补助金，标准为：七级伤残为13个月的本人工资，八级伤残为11个月的本人工资，九级伤残为9个月的本人工资，十级伤残为7个月的本人工资；(二)劳动、聘用合同期满终止，或者职工本人提出解除劳动、聘用合同的，由工伤保险基金支付一次性工伤医疗补助金，由用人单位支付一次性伤残就业补助金。一次性工伤医疗补助金和一次性伤残就业补助金的具体标准由省、自治区、直辖市人民政府规定。

（2022）浙0206民初1164号:
（类案背景与基础法理）
随着社会发展，人们的身体条件及医疗保障不断提高，人们的平均寿命不断增长，现行劳动者的退休年龄偏低，相当一部分劳动者在达到退休年龄后，存在继续为原单位或其他单位提供劳动的情况，这类人员在劳动过程中发生人身损害后，其权利如何予以保护成为司法实践中的一个常见问题。劳动者在提供劳动过程中受伤，如何维护其自身权利，需要根据劳动者与用人单位之间的基础法律关系进行考虑，不同的法律关系，对于维权的途径及结果存在重大影响。根据劳动者与其用人单位之间的关系，法律规定了两种不同的赔偿方式。即按照民法典侵权责任编一般原则进行赔偿或按照劳动关系工伤认定相关规范进行工伤赔偿。
（法律条款适用）
结合具体法律规定来看，《中华人民共和国劳动合同法实施条例》第二十一条规定: “劳动者达到法定退休年龄的，劳动合同终止。”这属于劳动合同终止的法定条件，当劳动者达到规定的法定退休年龄后，劳动合同终止，与用人单位间劳动法律关系终止。根据上述规定，应当认为劳动者达到法定退休年龄之后返聘或至其他单位工作继续提供劳务的，不应再适用劳动法律调整，双方建立的是劳务合同关系。"""

    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        print(chunk)
        print("....")