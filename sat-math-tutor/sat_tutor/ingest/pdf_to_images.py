"""
PDF to Image Conversion Module
Uses pdf2image (poppler) to convert PDF pages to PNG/JPG images
"""

import os
from pathlib import Path
from typing import Optional

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

from .page_range import parse_page_range


class PDFConversionError(Exception):
    """Exception raised for errors during PDF conversion"""
    pass


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Retrieves the total number of pages in a PDF.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Total page count
    """
    try:
        from pdf2image.pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        return info.get('Pages', 0)
    except PDFInfoNotInstalledError:
        raise PDFConversionError(
            "Poppler is not installed. Please install poppler:\n"
            "  macOS: brew install poppler\n"
            "  Ubuntu: sudo apt-get install poppler-utils\n"
            "  Windows: Download and add to PATH"
        )
    except Exception as e:
        raise PDFConversionError(f"Failed to retrieve PDF info: {str(e)}")


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    pages: Optional[str] = "all",
    dpi: int = 300,
    fmt: str = "png"
) -> list[str]:
    """
    Converts PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output images
        pages: Page range string, e.g., "1-3,5" or "all"
        dpi: Image resolution (Dots Per Inch)
        fmt: Image format (png or jpg)
    
    Returns:
        List of paths to the generated images
    
    Raises:
        PDFConversionError: If conversion fails
    """
    pdf_path = os.path.abspath(pdf_path)
    
    if not os.path.exists(pdf_path):
        raise PDFConversionError(f"PDF file does not exist: {pdf_path}")
    
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total page count
    total_pages = get_pdf_page_count(pdf_path)
    if total_pages == 0:
        raise PDFConversionError("PDF file is empty or unreadable")
    
    # Parse page range
    page_list = parse_page_range(pages, total_pages)
    if not page_list:
        raise PDFConversionError(f"No valid pages specified. Total pages available: {total_pages}")
    
    image_paths = []
    
    try:
        # Convert page by page (memory efficient)
        for page_num in page_list:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
                fmt=fmt
            )
            
            if images:
                # Save image
                filename = f"page_{page_num:03d}.{fmt}"
                image_path = os.path.join(output_dir, filename)
                images[0].save(image_path)
                image_paths.append(image_path)
    
    except PDFInfoNotInstalledError:
        raise PDFConversionError(
            "Poppler is not installed. Please install poppler:\n"
            "  macOS: brew install poppler\n"
            "  Ubuntu: sudo apt-get install poppler-utils\n"
            "  Windows: Download and add to PATH"
        )
    except PDFPageCountError as e:
        raise PDFConversionError(f"Error reading PDF page count: {str(e)}")
    except Exception as e:
        raise PDFConversionError(f"PDF conversion failed: {str(e)}")
    
    return image_paths


def convert_pdf_batch(
    pdf_paths: list[str],
    base_output_dir: str,
    pages: str = "all",
    dpi: int = 300
) -> dict[str, list[str]]:
    """
    Converts multiple PDFs in batch.
    
    Args:
        pdf_paths: List of paths to PDF files
        base_output_dir: Base directory for output
        pages: Page range string
        dpi: Image resolution
    
    Returns:
        Dictionary mapping {pdf_path: [list_of_image_paths]}
    """
    results = {}
    
    for pdf_path in pdf_paths:
        pdf_name = Path(pdf_path).stem
        output_dir = os.path.join(base_output_dir, pdf_name)
        
        try:
            images = pdf_to_images(pdf_path, output_dir, pages, dpi)
            results[pdf_path] = images
        except PDFConversionError as e:
            print(f"Warning: Conversion failed for {pdf_path}: {e}")
            results[pdf_path] = []
    
    return results