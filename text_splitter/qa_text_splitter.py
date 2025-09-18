# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class qatextsplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = ["？"]
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
    text_splitter = qatextsplitter(
        separators=["？"],
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=200,
        chunk_overlap=0
    )

    text = """1、误工费计算方式怎么确定？
误工费计算方式按受害者有无固定收入分为两种。
①固定收入，是指在国家机关、企事业单位、社会团体工作的人员本应按期得到的、却因事故造成耽误工作而丧失的工资、奖金、津贴、特殊工种的补助费等合法收入。一般以单位出具的收入证明和工资表为准，奖金以受害者上年度的单位人均奖金计算，超过奖金税起征点的以起征点为限。需要注意的是，个人独资、合伙企业等私营企业以及财务不健全的有限公司出具的特别是证明受害者“固定收入”高于上年度职工年平均工资3倍以上的收入证明，不到单独作为认定依据，须结合税务机关的个人所得税纳税证明等材料方能认定。
②无固定收入包括两类人员，一是从事农、林、牧、渔业生产的农村村民;二是有街道办事处、乡镇人民政府或者有关凭证，在事故发生前从事某种劳动，其收入能维持本人正常生活的，包括承包经营户、城乡个体工商户、打工者(散工、短工、临工)、家庭劳动服务人员等，均按事故发生地上一年度职工年平均工资计算。
此外，受害者依法从事第二职业的，其实际减少的收入，应予以合理赔偿。
受害者系未成年人等本身无劳动收入而要求赔偿误工费的，不予支持。
2、误工日期如何确定？
误工日期由受害者的住院天数和出院后治疗医院出具证明的休养天数两部分组成，从事故发生的当日开始计算，遇国家法定节假日均不扣减。治疗终结后无正当理由拒不出院或无相关证明擅自休养的，不予计算误工费。事故造成受害者残疾的，残疾者定残之后不再赔偿误工费。"""

    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        print(chunk)
        print("....")