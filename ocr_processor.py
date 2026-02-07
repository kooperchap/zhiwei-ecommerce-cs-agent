import fitz
import logging
import gc
from typing import List, Dict
from config import settings
from llms import get_llm

logger = logging.getLogger(__name__)

def split_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    滑动窗口切片：保证语义连续性
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        # 移动步长 = 窗口大小 - 重叠部分
        start += (chunk_size - overlap)
        
    return chunks

def ocr_remote(image_bytes: bytes) -> str:
    """
    仅在必要时调用大模型视觉能力
    """
    try:
        llm = get_llm()
        # 提示词专门优化为提取结构化内容
        prompt = "请详细描述这张图片的内容。如果是文档，请完整提取其中的文字；如果是图表，请描述其结构、数据和结论。"
        return llm.call_with_image(prompt, image_bytes)
    except Exception as e:
        logger.warning(f"OCR调用异常: {e}")
    return ""

def process_pdf(pdf_bytes: bytes, filename: str, category: str) -> List[Dict]:
    chunks_data = []
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = len(doc)
        logger.info(f"PDF解析开始 文件:{filename} 页数:{total}")
        
        for idx in range(total):
            page = doc[idx]
            # 策略1：优先尝试提取文本（极低成本，速度快）
            text = page.get_text().strip()
            source_type = "text_layer"
            
            # 策略2：如果文本极少（通常是扫描件或纯图），回退到视觉大模型
            if len(text) < 50:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img_bytes = pix.tobytes("png")
                    pix = None # 释放内存
                    
                    logger.info(f"第{idx+1}页为图片/扫描件，调用视觉模型解析...")
                    text = ocr_remote(img_bytes)
                    source_type = "visual_llm"
                    img_bytes = None
                except Exception as e:
                    logger.warning(f"第{idx+1}页视觉解析失败: {e}")

            if len(text) > 10:
                # 语义切片：带重叠的滑动窗口
                sub_chunks = split_text_with_overlap(text, chunk_size=500, overlap=100)
                
                for sub_text in sub_chunks:
                    chunks_data.append({
                        "content": sub_text,
                        "metadata": {
                            "source": filename,
                            "page": idx + 1,
                            "category": category,
                            "extract_mode": source_type
                        }
                    })
            
            if idx % 5 == 0:
                gc.collect()

    except Exception as e:
        logger.error(f"PDF处理异常: {e}")
    finally:
        if doc: doc.close()
        gc.collect()
        
    logger.info(f"PDF处理结束 生成切片数: {len(chunks_data)}")
    return {"status": "success", "docs": chunks_data}

def process_document(content: bytes, filename: str, category: str) -> Dict:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return process_pdf(content, filename, category)
    
    # 其他格式处理保持简单逻辑
    text = ""
    if fn.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    elif fn.endswith((".png", ".jpg", ".jpeg")):
        text = ocr_remote(content)
        
    if text:
        # 文本文件同样应用滑动窗口
        chunks = split_text_with_overlap(text, chunk_size=500, overlap=100)
        docs = [{"content": c, "metadata": {"source": filename, "category": category}} for c in chunks]
        return {"status": "success", "docs": docs}
        
    return {"status": "skipped", "docs": []}
