import os
import logging
import sys
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    data_dir = "test_data"
    if not os.path.exists(data_dir):
        logger.error(f"目录不存在 {data_dir}")
        return
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    logger.info(f"发现 {len(pdf_files)} 个 PDF 文件")
    if not pdf_files:
        return
    from ocr_processor import process_document
    from vectorstore import add_documents
    from es_client import bulk_index
    total_chunks = 0
    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)
        logger.info(f"正在处理: {filename}")
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            result = process_document(content, filename, "manual")
            if result["status"] == "success" and result["docs"]:
                docs_obj = [Document(page_content=d["content"], metadata=d["metadata"]) for d in result["docs"]]
                add_documents(docs_obj)
                try:
                    es_docs = [{"content": d["content"], **d["metadata"]} for d in result["docs"]]
                    bulk_index(es_docs)
                except Exception as e:
                    logger.warning(f"ES写入跳过: {e}")
                count = len(docs_obj)
                total_chunks += count
                logger.info(f"成功导入 {filename}: {count} 条")
            else:
                logger.warning(f"文件无有效内容: {filename}")
        except Exception as e:
            logger.error(f"导入失败 {filename}: {e}")
    logger.info(f"导入完成 总计: {total_chunks} 条")


if __name__ == "__main__":
    main()
