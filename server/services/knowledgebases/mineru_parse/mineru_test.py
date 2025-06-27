import os
import re
import tempfile
import shutil
import json
from typing import Callable, Tuple
from loguru import logger
from .minio_server import get_image_url
from .file_converter import ensure_pdf
from concurrent.futures import ThreadPoolExecutor

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

# 常量定义
OUTPUT_DIR_NAME = "output"
IMAGE_DIR_NAME = "images"
MINERU_BACKEND = "sglang-client"


def get_configured_mineru_server_url():
    """获取配置的mineru服务端地址"""
    return os.environ.get("MINERU_SERVER_URL", "http://127.0.0.1:30000")


def _setup_directories(base_job_temp_dir: str) -> Tuple[str, str]:
    """在指定的任务根临时目录下初始化 'output' 和 'output/images' 子目录。
    Args:
        base_job_temp_dir: 当前任务的专属根临时目录。
    Returns:
        A tuple: (full_path_to_images_dir, full_path_to_output_dir)
    """
    output_dir_path = os.path.join(base_job_temp_dir, OUTPUT_DIR_NAME)
    images_dir_path = os.path.join(output_dir_path, IMAGE_DIR_NAME)
    os.makedirs(images_dir_path, exist_ok=True)
    logger.info(f"Ensured output directory exists: {output_dir_path}")
    logger.info(f"Ensured images directory exists: {images_dir_path}")
    return images_dir_path, output_dir_path

def _read_pdf_bytes(pdf_file_path: str) -> bytes:
    """读取PDF文件为字节流"""
    logger.debug(f"Reading PDF bytes from: {pdf_file_path}")
    reader = FileBasedDataReader("")
    return reader.read(pdf_file_path)

def _process_pdf_content(pdf_bytes: bytes, images_full_path: str) -> any:
    """处理PDF内容
    
    Args:
        pdf_bytes: PDF文件的二进制内容
        images_full_path: 图片写入器将使用的绝对基础路径 (例如 /tmp/jobXYZ/output/images)
    
    Returns:
        middle_json: 处理后的json结果对象
    """
    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
    
    image_writer = FileBasedDataWriter(images_full_path)
    server_url = get_configured_mineru_server_url()
    
    with ThreadPoolExecutor() as executor:
            future = executor.submit(vlm_doc_analyze, pdf_bytes, image_writer=image_writer, backend=MINERU_BACKEND, server_url=server_url)
            middle_json, _ = future.result()
    
    return middle_json

def _generate_markdown(middle_json: any, name_without_suff: str, output_full_path: str, images_subdir_name_for_ref: str) -> str:
    """生成Markdown文件及相关内容"""
    md_writer = FileBasedDataWriter(output_full_path)
    md_file_name = f"{name_without_suff}.md"
    md_file_path = os.path.join(output_full_path, md_file_name)
    logger.info(f"生成Markdown文件: {md_file_path}")
    
    pdf_info = middle_json["pdf_info"]
    md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, images_subdir_name_for_ref)
    md_writer.write_string(md_file_name, md_content_str )
    
    content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, images_subdir_name_for_ref)
    md_writer.write_string(f"{name_without_suff}_content_list.json", json.dumps(content_list, ensure_ascii=False, indent=4))

    md_writer.write_string(f"{name_without_suff}_middle.json", json.dumps(middle_json, ensure_ascii=False, indent=4))

    return md_file_path

def process_pdf_with_minerU(file_input, update_progress=None):
    """
    处理PDF文件并生成Markdown
    Args:
        file_input: 可以是PDF文件路径、URL或Office文件路径
        update_progress: 进度回调函数
    Returns:
        str: 生成的Markdown文件路径
    """
    # 使用当前目录作为工作目录
    job_specific_temp_dir = os.getcwd()
    
    # 确保PDF文件存在
    pdf_path_to_process, path_to_delete_after_processing = ensure_pdf(file_input, job_specific_temp_dir)

    try:
        update_progress(0.1, "=== 开始文件预处理 ===")
        logger.info(f"接收到输入进行处理: {file_input}")

        if not pdf_path_to_process:
            logger.error(f"无法获取或生成用于处理的PDF文件，源输入: {file_input}")
            raise Exception(f"无法处理输入 {file_input}，未能转换为PDF或找到PDF。")
        
        logger.info(f"将要处理的PDF文件: {pdf_path_to_process}")
        if path_to_delete_after_processing:
            logger.info(f"此PDF是临时转换文件，将在处理后删除: {path_to_delete_after_processing}")

        update_progress(0.3, f"PDF文件准备就绪 ({os.path.basename(pdf_path_to_process)})，开始MinerU核心处理...")
        
        images_full_path, output_full_path = _setup_directories(job_specific_temp_dir)
        
        name_without_suff = os.path.splitext(os.path.basename(pdf_path_to_process))[0]
        pdf_bytes = _read_pdf_bytes(pdf_path_to_process)
        
        middle_json = _process_pdf_content(pdf_bytes, images_full_path)
        
        md_file_path = _generate_markdown(middle_json, name_without_suff, output_full_path, IMAGE_DIR_NAME)
        
        update_progress(0.5, f"文件处理和Markdown生成完成: {os.path.basename(md_file_path)}")
        logger.info(f"最终生成的Markdown文件路径: {md_file_path}")
        
        return md_file_path

    except Exception as e:
        logger.exception(f"在 process_pdf_with_minerU 中发生严重错误，输入: {file_input}, 错误: {e}")
        update_progress(0.5, f"文件处理失败: {str(e)}")
        raise
    finally:
        if path_to_delete_after_processing and os.path.exists(path_to_delete_after_processing):
            try:
                os.remove(path_to_delete_after_processing)
                logger.info(f"已清理临时转换的PDF文件: {path_to_delete_after_processing}")
            except OSError as e_remove:
                logger.error(f"清理临时转换的PDF文件失败: {path_to_delete_after_processing}, 错误: {e_remove}")
        
        if job_specific_temp_dir and os.path.exists(job_specific_temp_dir):
            logger.info(f"任务专属临时目录 {job_specific_temp_dir} 及其内容 (包括最终输出的 markdown 和图片) 被保留。调用者负责后续清理。")
            pass


def update_markdown_image_urls(md_file_path, kb_id):
    """更新Markdown文件中的图片URL"""
    def _replace_img(match):
        img_url = os.path.basename(match.group(1))
        if not img_url.startswith(('http://', 'https://')):
            img_url = get_image_url(kb_id, img_url)
        return f'<img src="{img_url}" style="max-width: 300px;" alt="图片">'
    with open(md_file_path, 'r+', encoding='utf-8') as f:
        content = f.read()
        updated_content = re.sub(r'!\[\]\((.*?)\)', _replace_img, content)
        f.seek(0)
        f.write(updated_content)
        f.truncate()
    return updated_content