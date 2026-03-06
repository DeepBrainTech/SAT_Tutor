"""
Page Range Parsing Module
Supported formats: "1-3,5,7-10" or "all"
"""

from typing import Optional


def parse_page_range(page_str: str, total_pages: int) -> list[int]:
    """
    Parses a page range string into a list of integers.
    
    Args:
        page_str: Page range string, e.g., "1-3,5,7-10" or "all"
        total_pages: Total number of pages in the document
    
    Returns:
        List of page numbers (1-indexed)
    
    Examples:
        >>> parse_page_range("1-3,5", 10)
        [1, 2, 3, 5]
        >>> parse_page_range("all", 5)
        [1, 2, 3, 4, 5]
    """
    page_str = page_str.strip().lower()
    
    if page_str == "all":
        return list(range(1, total_pages + 1))
    
    pages = set()
    
    for part in page_str.split(","):
        part = part.strip()
        if not part:
            continue
            
        if "-" in part:
            # Range format: "1-3"
            try:
                start, end = part.split("-", 1)
                start = int(start.strip())
                end = int(end.strip())
                
                # Boundary checks
                start = max(1, start)
                end = min(total_pages, end)
                
                if start <= end:
                    pages.update(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid page range: {part}")
        else:
            # Single page number
            try:
                page = int(part)
                if 1 <= page <= total_pages:
                    pages.add(page)
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")
    
    return sorted(pages)


def validate_page_range(page_str: str) -> bool:
    """
    Validates if the page range string format is correct.
    
    Args:
        page_str: Page range string
    
    Returns:
        True if valid, False otherwise
    """
    page_str = page_str.strip().lower()
    
    if page_str == "all":
        return True
    
    for part in page_str.split(","):
        part = part.strip()
        if not part:
            continue
            
        if "-" in part:
            segments = part.split("-", 1)
            if len(segments) != 2:
                return False
            try:
                int(segments[0].strip())
                int(segments[1].strip())
            except ValueError:
                return False
        else:
            try:
                int(part)
            except ValueError:
                return False
    
    return True


def format_page_range(pages: list[int]) -> str:
    """
    Formats a list of page numbers into a condensed range string.
    
    Args:
        pages: List of page numbers
    
    Returns:
        Formatted range string
    
    Examples:
        >>> format_page_range([1, 2, 3, 5, 7, 8, 9])
        "1-3, 5, 7-9"
    """
    if not pages:
        return ""
    
    pages = sorted(set(pages))
    ranges = []
    start = pages[0]
    end = pages[0]
    
    for page in pages[1:]:
        if page == end + 1:
            end = page
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = page
            end = page
    
    # Handle the last remaining range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    return ", ".join(ranges)