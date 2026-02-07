import logging
from typing import Optional
from llms import get_llm

logger = logging.getLogger(__name__)


class MultimodalProcessor:
    def __init__(self):
        self.llm = get_llm()

    def process_image(self, image_data: bytes, question: str = None) -> str:
        prompt = question if question else "请描述这张图片的内容，如果包含文字请提取出来。"
        try:
            result = self.llm.call_with_image(prompt, image_data)
            return result
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            return "图片处理失败，请重试"

    def extract_text_from_image(self, image_data: bytes) -> str:
        prompt = "请提取这张图片中的所有文字内容，保持原有格式，只返回文字不要其他说明。"
        try:
            result = self.llm.call_with_image(prompt, image_data)
            return result.strip()
        except Exception as e:
            logger.error(f"OCR提取失败: {e}")
            return ""


_processor: Optional[MultimodalProcessor] = None


def get_processor() -> MultimodalProcessor:
    global _processor
    if _processor is None:
        _processor = MultimodalProcessor()
    return _processor
