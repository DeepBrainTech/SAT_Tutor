"""
OCR Text Extraction Module
Extracts text from images using pytesseract
"""

import os
from typing import Optional
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from ..utils.logging import Logger


class OCRExtractor:
    """
    OCR Text Extractor
    Extracts text from images using Tesseract OCR
    """
    
    def __init__(self, lang: str = "eng", logger: Optional[Logger] = None):
        """
        Initialize OCR extractor
        
        Args:
            lang: OCR language (default: eng for English)
            logger: Logger instance
        """
        if not OCR_AVAILABLE:
            raise ImportError(
                "pytesseract and Pillow are required for OCR. "
                "Install with: pip install pytesseract Pillow\n"
                "Also install Tesseract: brew install tesseract (macOS) "
                "or apt-get install tesseract-ocr (Linux)"
            )
        
        self.lang = lang
        self.logger = logger
    
    def _log(self, message: str, level: str = "info"):
        """Log a message"""
        if self.logger:
            self.logger.log(message, level)
    
    def extract_text_from_image(self, image_path: str) -> tuple[str, Optional[str]]:
        """
        Extract text from a single image
        
        Args:
            image_path: Path to the image file
        
        Returns:
            (extracted_text, error_message or None)
        """
        if not os.path.exists(image_path):
            return "", f"Image file does not exist: {image_path}"
        
        try:
            image = Image.open(image_path)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(
                image, 
                lang=self.lang,
                config='--psm 6'  # Assume uniform block of text
            )
            
            return text.strip(), None
            
        except Exception as e:
            error_msg = f"OCR extraction failed for {image_path}: {str(e)}"
            self._log(error_msg, "error")
            return "", error_msg
    
    def extract_text_from_images(
        self,
        image_paths: list[str],
        start_page: int = 1
    ) -> tuple[dict[int, str], list[int], list[str]]:
        """
        Extract text from multiple images
        
        Args:
            image_paths: List of image paths
            start_page: Starting page number
        
        Returns:
            (dict of page_number -> text, list of failed pages, list of errors)
        """
        all_texts = {}
        failed_pages = []
        errors = []
        
        for i, image_path in enumerate(image_paths):
            # Extract page number from filename
            page_num = start_page + i
            filename = Path(image_path).stem
            if filename.startswith("page_"):
                try:
                    page_num = int(filename.split("_")[1])
                except (IndexError, ValueError):
                    pass
            
            self._log(f"Extracting text from page {page_num}...")
            
            text, error = self.extract_text_from_image(image_path)
            
            if text:
                all_texts[page_num] = text
                self._log(f"Extracted {len(text)} characters from page {page_num}")
            
            if error:
                failed_pages.append(page_num)
                errors.append(error)
        
        return all_texts, failed_pages, errors
    
    def combine_texts(self, page_texts: dict[int, str]) -> str:
        """
        Combine texts from multiple pages into one string
        
        Args:
            page_texts: Dict of page_number -> text
        
        Returns:
            Combined text with page markers
        """
        combined = []
        for page_num in sorted(page_texts.keys()):
            text = page_texts[page_num]
            combined.append(f"=== PAGE {page_num} ===\n{text}")
        
        return "\n\n".join(combined)

