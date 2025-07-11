import os

from marker.scripts.common import (
    load_models,
    parse_args,
    img_to_html,
    get_page_image,
    page_count,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"

from marker.settings import settings
from streamlit.runtime.uploaded_file_manager import UploadedFile

import re
import tempfile
from typing import Any, Dict

import streamlit as st
from PIL import Image

from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from marker.scripts.upload_notion import create_notion_page_from_md, insert_md_to_page, insert_md_to_page_by_doi
import re

def convert_pdf(fname: str, config_parser: ConfigParser) -> (str, Dict[str, Any], dict):
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1
    converter_cls = PdfConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    return converter(fname)


def markdown_insert_images(markdown, images):
    image_tags = re.findall(
        r'(!\[(?P<image_title>[^\]]*)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))',
        markdown,
    )

    for image in image_tags:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if image_path in images:
            markdown = markdown.replace(
                image_markdown, img_to_html(images[image_path], image_alt)
            )
    return markdown

def get_ollama_models():
    import requests

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
    response = requests.get(f"{ollama_base_url}/api/tags")
    if response.status_code == 200:
        return [model["name"] for model in response.json()["models"]]
    else:
        st.error("Failed to fetch Ollama models. Please check the base URL.")
        return [os.environ.get("OLLAMA_MODEL")]

def zip_markdown(markdown: str, images: Dict[str, Any]) -> bytes:
    import zipfile
    from io import BytesIO
    from PIL import Image

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr(in_file.name.split('.')[0] + ".md", markdown)
        for image_name, image_data in images.items():
            if isinstance(image_data, Image.Image):
                img_bytes = BytesIO()
                if image_name.endswith(".png"):
                    image_data.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    zip_file.writestr(image_name, img_bytes.read())
                elif image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
                    image_data.save(img_bytes, format="JPEG")
                    img_bytes.seek(0)
                    zip_file.writestr(image_name, img_bytes.read())
                else:
                    raise ValueError(f"Unsupported image format for {image_name}")
            else:
                zip_file.writestr(image_name, image_data)
    return zip_buffer.getvalue()

@st.dialog("Upload to Notion")
def upload_to_notion(text: str, images: Dict[str, Any], doi: str | None = None, replace_existing: bool = True):
    with st.spinner("Uploading to Notion..."):
        result = insert_md_to_page_by_doi(doi, text, images, replace_existing=replace_existing)
        if result:
            st.success("Markdown uploaded to Notion successfully!")
        else:
            st.error("Failed to upload Markdown to Notion.")


st.set_page_config(layout="wide", page_title="Marker Extraction")
st.sidebar.title("Marker Extraction")
col1, col2 = st.columns([0.5, 0.5])

model_dict = load_models()
cli_options = parse_args()
# st.markdown("""
# # Marker Demo

# This app will let you try marker, a PDF or image -> Markdown, HTML, JSON converter. It works with any language, and extracts images, tables, equations, etc.

# Find the project [here](https://github.com/VikParuchuri/marker).
# """)

in_file: UploadedFile = st.sidebar.file_uploader(
    "PDF, document, or image file:",
    type=["pdf", "png", "jpg", "jpeg", "gif", "pptx", "docx", "xlsx", "html", "epub"],
)

if in_file is None:
    st.stop()
filetype = in_file.type

with col1:
    page_count = page_count(in_file)
    page_number = st.number_input(
        f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count
    )
    pil_image = get_page_image(in_file, page_number)
    st.image(pil_image, use_container_width=True)

page_range = st.sidebar.text_input(
    "Page range to parse, comma separated like 0,5-10,20",
    value=f"{page_number}-{page_number}",
)
output_format = st.sidebar.selectbox(
    "Output format", ["markdown", "json", "html", "chunks"], index=0
)

use_llm = st.sidebar.checkbox(
    "Use LLM", help="Use LLM for higher quality processing", value=False
)
ollama_model = ""
ollama_base_url = ""
llm_service = ""

if use_llm:
    llm_service = st.sidebar.selectbox(
        "LLM service",
        ["ollama", "openai", "azure_openai", "anthropic", "local"],
        index=0,
    )
    if llm_service == "ollama":
        ollama_model = st.sidebar.selectbox(
            "Ollama model",
            get_ollama_models(),
            index=0,
            help="Select the Ollama model to use for LLM processing",
        )
        ollama_base_url = st.sidebar.text_input(
            "Ollama base URL",
            value=os.environ.get("OLLAMA_BASE_URL"),
            help="Base URL for Ollama LLM service",
        )

force_ocr = st.sidebar.checkbox("Force OCR", help="Force OCR on all pages", value=False)
strip_existing_ocr = st.sidebar.checkbox(
    "Strip existing OCR",
    help="Strip existing OCR text from the PDF and re-OCR.",
    value=False,
)
debug = st.sidebar.checkbox("Debug", help="Show debug information", value=False)
format_lines = st.sidebar.checkbox(
    "Format lines",
    help="Format lines in the document with OCR model",
    value=False,
)
disable_ocr_math = st.sidebar.checkbox(
    "Disable math",
    help="Disable math in OCR output - no inline math",
    value=False,
)

@st.dialog("Run Extraction")
def run_marker_action():
    with st.spinner("Running Marker..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_pdf = os.path.join(tmp_dir, "temp.pdf")
            with open(temp_pdf, "wb") as f:
                f.write(in_file.getvalue())

            cli_options.update(
                {
                    "output_format": output_format,
                    "page_range": page_range,
                    "force_ocr": force_ocr,
                    "debug": debug,
                    "output_dir": settings.DEBUG_DATA_FOLDER if debug else None,
                    "use_llm": use_llm,
                    "strip_existing_ocr": strip_existing_ocr,
                    "format_lines": format_lines,
                    "disable_ocr_math": disable_ocr_math,
                    "ollama_base_url": ollama_base_url,
                    "ollama_model": ollama_model,
                    "llm_service": f"marker.services.{llm_service}.{llm_service.capitalize()}Service",
                }
            )
            config_parser = ConfigParser(cli_options)
            rendered = convert_pdf(temp_pdf, config_parser)
            page_range_val = config_parser.generate_config_dict()["page_range"]
            first_page = page_range_val[0] if page_range_val else 0
        text, ext, images = text_from_rendered(rendered)
        st.session_state["text"] = text
        st.session_state["images"] = images
        st.session_state["output_format"] = output_format
        st.session_state["rendered_obj"] = rendered
        st.session_state["first_page_obj"] = first_page
        st.success("Extraction completed successfully!")

run_marker = st.sidebar.button("Run Marker", on_click=run_marker_action)

if "text" in st.session_state:
    with col2:
        if st.session_state["output_format"] == "markdown":
            html_text = markdown_insert_images(st.session_state["text"], st.session_state["images"])
            st.markdown(html_text, unsafe_allow_html=True)
        elif st.session_state["output_format"] == "json":
            st.json(st.session_state["text"])
        elif st.session_state["output_format"] == "html":
            st.html(st.session_state["text"])
        elif st.session_state["output_format"] == "chunks":
            st.json(st.session_state["text"])

    if debug:
        with col1:
            debug_data_path = st.session_state["rendered_obj"].metadata.get("debug_data_path")
            if debug_data_path:
                pdf_image_path = os.path.join(debug_data_path, f"pdf_page_{st.session_state['first_page_obj']}.png")
                img = Image.open(pdf_image_path)
                st.image(img, caption="PDF debug image", use_container_width=True)
                layout_image_path = os.path.join(
                    debug_data_path, f"layout_page_{st.session_state['first_page_obj']}.png"
                )
                img = Image.open(layout_image_path)
                st.image(img, caption="Layout debug image", use_container_width=True)
            st.write("Raw output:")
            st.code(st.session_state["text"], language=st.session_state["output_format"])

@st.fragment
def download_markdown():
    st.download_button(
            "Download Markdown",
            st.session_state["text"],
            file_name=f"{in_file.name.split('.')[0]}.md",
            mime="text/markdown",
        )
    st.download_button(
            "Download zipped Markdown",
            data=zip_markdown(st.session_state["text"], st.session_state["images"]),
            file_name=f"{in_file.name.split('.')[0]}.zip",
            mime="application/zip",
        )
    
    doi = st.text_input(
        "DOI",
        value="",
        help="Enter the DOI of the document to link in Notion",
    )

    replace_or_append = st.radio(
        "Replace or append to existing Notion page",
        ["Replace", "Append"],
        index=0,
        help="Choose whether to replace the existing Notion page or append to it.",
        disabled= not doi,
    )
    
    st.button(
        "Upload to Notion",
        on_click=upload_to_notion,
        args=(st.session_state["text"], st.session_state["images"], doi, replace_or_append),
        help="Upload the extracted text to Notion",
        disabled= not doi
    )

with st.sidebar:
    if "text" in st.session_state:
        download_markdown()
