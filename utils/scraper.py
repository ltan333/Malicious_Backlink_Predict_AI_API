# utils/scraper.py
import httpx
import fitz
from urllib.parse import urlparse
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

def is_accessible(url, timeout=5):
    """
    Check if a given URL is accessible (status code < 400).
    Sends a HEAD request using httpx.
    """
    try:
        response = httpx.head(url, timeout=timeout, follow_redirects=True, verify=False)
        return response.status_code < 400
    except:
        return False

def extract_pdf_text(pdf_bytes):
    """
    Extract plain text from a PDF file given its byte content.
    Uses PyMuPDF (fitz) to read the PDF pages.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    except:
        return ""

def fetch_pdf_httpx(url, timeout=10):
    """
    Attempt to fetch a PDF file from a URL using httpx.
    Returns the PDF bytes if valid, else None.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf",
        "Referer": url,
    }
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers, verify=False) as client:
            r = client.get(url)
            if r.status_code == 200 and r.content.startswith(b"%PDF"):
                return r.content
    except:
        pass
    return None

async def fetch_pdf_playwright(url, timeout=20):
    """
    Use Playwright to render the webpage and fetch PDF content if available.
    Useful if PDF is dynamically generated via JS.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(ignore_https_errors=True)
                page = await context.new_page()
                await page.goto(url, timeout=timeout * 1000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    await page.wait_for_timeout(4000)
                html = await page.content()
                return html
            finally:
                await browser.close()
    except:
        pass
    return None

async def handle_pdf(url):
    """
    Try to fetch PDF content from a URL using httpx or Playwright.
    Extract and return the text content, or 'inaccessible' if it fails.
    """
    pdf_data = fetch_pdf_httpx(url) or await fetch_pdf_playwright(url)
    if pdf_data:
        text = extract_pdf_text(pdf_data)
        return text if text else "inaccessible"
    return "inaccessible"

def fetch_static_html(url, timeout=10):
    """
    Fetch static HTML content of a webpage using httpx.
    Returns the raw HTML or an empty string on failure.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(follow_redirects=True, verify=False, timeout=timeout, headers=headers) as client:
            r = client.get(url)
            return r.text
    except:
        return ""

async def fetch_dynamic_html(url, timeout=10):
    """
    Fetch HTML content rendered dynamically (SPA) using Playwright.
    Waits for content to load before returning.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await (await browser.new_context(ignore_https_errors=True)).new_page()
            await page.goto(url, timeout=timeout * 1000)
            await page.wait_for_timeout(4000)
            html = await page.content()
            await browser.close()
            return html
    except:
        return ""

async def handle_html(url):
    """
    Extract clean text from HTML content.
    Fallbacks to dynamic rendering if site uses frameworks like React/Vue.
    Removes scripts, styles, and hidden elements.
    """
    static_html = fetch_static_html(url)
    if not static_html:
        return "inaccessible"

    soup = BeautifulSoup(static_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
        tag.decompose()

    text = soup.get_text("\n", strip=True)

    has_fw = any(f in static_html.lower() for f in ["vue", "react", "angular", "webpack", "require.js"])
    has_ns = "enable javascript" in static_html.lower()

    if has_fw or has_ns:
        dynamic_html = await fetch_dynamic_html(url)
        if not dynamic_html:
            return text or "inaccessible"
        soup = BeautifulSoup(dynamic_html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

    return text or "inaccessible"

async def get_website_context_async(url):
    """
    Unified function to return website's textual content.
    Decides whether the URL is a PDF or HTML and handles accordingly.
    """
    if not is_accessible(url):
        return "inaccessible"

    ext = Path(urlparse(url).path).suffix.lower()
    if not ext:
        ext = ".html"

    if ext == ".pdf":
        result = await handle_pdf(url)
    else:
        result = await handle_html(url)

    return result