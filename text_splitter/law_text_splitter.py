# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class lawtextsplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = ["第[零一二三四五六七八九十百千万亿]+条\s",]
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
    text_splitter = lawtextsplitter(
        separators=None,#["？"]
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=200,
        chunk_overlap=0
    )

    text = """第一条　为了加强音像制品的管理，促进音像业的健康发展和繁荣，丰富人民群众的文化生活，促进社会主义物质文明和精神文明建设，制定本条例。
第二条　本条例适用于录有内容的录音带、录像带、唱片、激光唱盘和激光视盘等音像制品的出版、制作、复制、进口、批发、零售、出租等活动。
音像制品用于广播电视播放的，适用广播电视法律、行政法规。
第三条　出版、制作、复制、进口、批发、零售、出租音像制品，应当遵守宪法和有关法律、法规，坚持为人民服务和为社会主义服务的方向，传播有益于经济发展和社会进步的思想、道德、科学技术和文化知识。
"""

    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        print(chunk)
        print("....")