# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import ssl
import os
import fitz
import httpx
import asyncio
import tldextract
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse, parse_qs, unquote
from playwright.async_api import async_playwright, Error as PlaywrightError
from typing import List, Optional, Set

from app.cores.logging import logging
from app.schemas.schemas import OutputEntry
from app.utils.preprocessing import preprocess_text, SENTINELS, PDF_EMPTY_SENTINEL, PDF_FETCH_SENTINEL, CLOUDFLARE_BLOCKED, NOT_FOUND, HTTP_5XX_SAFE, EXTERNAL_REDIRECT, INACCESSIBLE
from app.services.redirect_service import same_etld1, is_trusted_destination, extract_redirect_target

# --- CACHE ---
domain_dict = {}
NOT_FOUND_BACKLINKS: set[str] = set()

# --- PLAYWRIGHT ---
_playwright = None
_browser = None
_browser_lock = asyncio.Lock()
_static_client = None

# --- SCRAPER ---
def init_static_client():
    """Create a pooled HTTP client for static HTML (keep-alive + HTTP/2)."""
    global _static_client
    if _static_client is None:
        _static_client = httpx.Client(
            http2=True,
            follow_redirects=True,
            verify=make_weak_ssl_context(),
            limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
            timeout=httpx.Timeout(connect=1.0, read=3.0, write=3.0, pool=3.0),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )

def close_static_client():
    """Close and reset the pooled client."""
    global _static_client
    try:
        if _static_client is not None:
            _static_client.close()
    finally:
        _static_client = None

async def init_browser():
    """Launch a single Chromium instance and reuse it."""
    global _playwright, _browser
    async with _browser_lock:
        if _playwright is None:
            _playwright = await async_playwright().start()
        if _browser is None or not _browser.is_connected():
            _browser = await _playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
    return _browser

async def get_page():
    browser = await init_browser()
    context = await browser.new_context(
        ignore_https_errors=True,
        accept_downloads=True,
        locale="vi-VN",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
        ),
        extra_http_headers={
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        },
    )
    page = await context.new_page()
    return page, context

def _domain_key(u: str) -> str:
    host = urlparse(u).hostname or "default"
    try:
        ext = tldextract.extract(host)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else host
    except Exception:
        return host

async def get_page_with_state(url: str):
    browser = await init_browser()
    dom = _domain_key(url)
    # storage under app/ directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state_dir = os.path.join(base_dir, "storage_state")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, f"{dom}.json")

    storage_state_obj = None
    if os.path.exists(state_path) and os.path.getsize(state_path) > 2:
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                import json as _json
                storage_state_obj = _json.load(f)
        except Exception:
            storage_state_obj = None
    context = await browser.new_context(
        ignore_https_errors=True,
        accept_downloads=True,
        locale="vi-VN",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        extra_http_headers={
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        },
        storage_state=storage_state_obj,
    )
    await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
    page = await context.new_page()
    return page, context, state_path

async def close_browser():
    """Close the shared browser on app shutdown."""
    global _browser, _playwright
    try:
        if _browser and _browser.is_connected():
            await _browser.close()
    finally:
        _browser = None
        if _playwright:
            await _playwright.stop()
            _playwright = None
        close_static_client()

# --- CHECK URL ---
async def is_accessible(url: str, timeout: int = 8) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": _origin(url),
    }
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            verify=make_weak_ssl_context(),
            timeout=timeout,
            headers=headers,
        ) as client:
            try:
                r = await client.get(url)
                s = r.status_code or 0
                if s in (404, 410):
                    return False
                if 500 <= s < 600:
                    return True
                if (200 <= s < 400) or s in (401, 403):
                    return True
            except httpx.HTTPError:
                pass

            try:
                rh = await client.head(url)
                s = rh.status_code or 0
                if s in (404, 410):
                    return False
                if 500 <= s < 600:
                    return True
                return (200 <= s < 400) or s in (401, 403)
            except httpx.HTTPError:
                return False
    except Exception:
        return False

# --- EXTRACT PDF TEXT ---
def extract_pdf_text(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for page in doc:
            for mode in ("text", "raw", "blocks", "xhtml"):
                t = page.get_text(mode)
                if t and t.strip():
                    texts.append(t)
                    break
            else:
                pass
        return "\n".join(texts).strip()
    except Exception as e:
        logging.debug(f"extract_pdf_text failed: {e}")
        return ""
    
# --- FETCH PDF HTTPX---
def _origin(u):
    p = urlparse(u)
    return urlunparse((p.scheme, p.netloc, "/", "", "", ""))

def make_weak_ssl_context() -> ssl.SSLContext:
    """
    Allow legacy TLS/cert signatures (SECLEVEL=0) and legacy handshakes.
    Only use for fetching public files from legacy servers.
    """
    ctx = ssl.create_default_context()
    if hasattr(ssl, "OP_LEGACY_SERVER_CONNECT"):
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
    try:
        ctx.set_ciphers("ALL:@SECLEVEL=0")
    except ssl.SSLError:
        pass
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_pdf_httpx(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": _origin(url)
    }
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
            verify=make_weak_ssl_context()
        ) as client:
            r = client.get(url)
            status = r.status_code
            ctype  = (r.headers.get("content-type") or "").lower()
            head   = r.content[:65536]

            if status in (404, 410):
                return NOT_FOUND
            if 500 <= status < 600:
                return HTTP_5XX_SAFE

            if "text/html" in ctype and is_cloudflare_html(head.decode(errors="ignore")):
                logging.info(f"[cloudflare] pdf path challenged for {url}")
                return CLOUDFLARE_BLOCKED.encode()

            disp = (r.headers.get("content-disposition") or "").lower()
            looks_like_pdf = (
                "application/pdf" in ctype
                or ("filename=" in disp and ".pdf" in disp)
                or urlparse(url).path.lower().endswith(".pdf")
                or (b"%PDF" in head)
            )
            if status == 200 and looks_like_pdf:
                return r.content
    except Exception as e:
        logging.info(f"[pdf] httpx get failed for {url}: {e}")
    return None

# --- HANDLE PDF ---
async def handle_pdf(url):
    try:
        pdf_bytes = fetch_pdf_httpx(url)

        if pdf_bytes == HTTP_5XX_SAFE:
            return HTTP_5XX_SAFE
        if pdf_bytes == NOT_FOUND:
            return NOT_FOUND
        if pdf_bytes == PDF_FETCH_SENTINEL:
            return PDF_FETCH_SENTINEL
        if pdf_bytes == CLOUDFLARE_BLOCKED.encode():
            return CLOUDFLARE_BLOCKED

        if not pdf_bytes:
            logging.info(f"[pdf] fetch failed for {url} -> trying Playwright")
            pdf_bytes = await fetch_pdf_playwright(url)

        if not pdf_bytes:
            logging.info(f"[pdf] both httpx and Playwright failed for {url}")
            return PDF_FETCH_SENTINEL

        if pdf_bytes == CLOUDFLARE_BLOCKED.encode():
            return CLOUDFLARE_BLOCKED

        text = extract_pdf_text(pdf_bytes)
        if text and text.strip():
            logging.info(f"[pdf] extracted_text_len={len(text)}")
            return preprocess_text(text)

        logging.info(f"[pdf] accessible but empty for {url}")
        return PDF_EMPTY_SENTINEL

    except Exception as e:
        logging.debug(f"handle_pdf failed for {url}: {e}")
        return PDF_FETCH_SENTINEL

# --- FETCH STATIC HTML ---
def fetch_static_html(url):
    try:
        init_static_client()
        r = _static_client.get(url)
        if r.history:
            final_url = str(r.url)
            if not same_etld1(url, final_url) and not is_trusted_destination(final_url):
                return EXTERNAL_REDIRECT
            if r.status_code in (404, 410):
                return NOT_FOUND
        if 500 <= (r.status_code or 0) < 600:
            return HTTP_5XX_SAFE
            return r.text
    except Exception:
        return ""

# --- SECOND-PASS: LIGHTWEIGHT ACCESS PROBE (no Playwright, no PDF parsing) ---
async def access_probe(url: str) -> str:
    try:
        origin = _origin(url)
        # Use a short-lived client to avoid cross-thread calls to the pooled client
        def _probe():
            try:
                with httpx.Client(
                    http2=True,
                    follow_redirects=True,
                    verify=make_weak_ssl_context(),
                    timeout=httpx.Timeout(connect=1.5, read=4.0, write=4.0, pool=4.0),
                    headers={
                        "User-Agent": _static_client.headers.get("User-Agent", "Mozilla/5.0") if _static_client else "Mozilla/5.0",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                        "Referer": origin,
                    }
                ) as client:
                    r = client.get(url)
                    if r.history:
                        final_url = str(r.url)
                        if not same_etld1(url, final_url) and not is_trusted_destination(final_url):
                            return EXTERNAL_REDIRECT
                    if r.status_code in (404, 410):
                        return NOT_FOUND
                    if 500 <= (r.status_code or 0) < 600:
                        return HTTP_5XX_SAFE
                    text = r.text[:4000]
                    return CLOUDFLARE_BLOCKED if is_cloudflare_html(text) else "ok"
            except Exception:
                return INACCESSIBLE
        return await asyncio.to_thread(_probe)
    except Exception:
        return INACCESSIBLE

# --- CLOUDFLARE DETECTION ---
_CF_MARKERS = (
    "attention required! | cloudflare",
    "ddos protection by cloudflare",
    "checking your browser before accessing",
    "please stand by, while we are checking your browser",
    "cf-chl-",
    "cf-error-",
    "cf-ray",
    "just a moment...",
)

_CF_MARKERS_VI = (
    "vui lòng đợi",
    "chờ một chút",
    "chờ chút",
    "đang kiểm tra trình duyệt của bạn",
    "chúng tôi đang kiểm tra trình duyệt",
    "đang xác minh",
    "vui lòng đứng yên",
    "chỉ một lát", "chờ trong giây lát",
)

def is_cloudflare_html(html: str) -> bool:
    if not html:
        return False
    low = html.lower()
    return any(m in low for m in (_CF_MARKERS + _CF_MARKERS_VI))

# --- FETCH DYNAMIC HTML ---
playwright_semaphore = asyncio.Semaphore(3)

async def fetch_dynamic_html(url: str, timeout: int = 12) -> str:
    """
    Fetch page content using Playwright, waiting for dynamic data.
    Returns one of: EXTERNAL_REDIRECT, HTTP_5XX_SAFE, NOT_FOUND, CLOUDFLARE_BLOCKED, INACCESSIBLE, or raw HTML.
    """
    page = None
    context = None
    start = asyncio.get_event_loop().time()

    def _ms_left():
        spent = asyncio.get_event_loop().time() - start
        left = max(0.5, timeout - spent)
        return int(left * 1000)

    async with playwright_semaphore:
        try:
            page, context, state_path = await get_page_with_state(url)

            nav_timeout = _ms_left()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
            status = (resp.status if resp else None)
            final_url = page.url or url

            if status in (404, 410):
                return NOT_FOUND
            if status and 500 <= status < 600:
                return HTTP_5XX_SAFE
            if status in (401, 403, 429, 451):
                return CLOUDFLARE_BLOCKED

            try:
                if not same_etld1(url, final_url) and not is_trusted_destination(final_url):
                    return EXTERNAL_REDIRECT
            except Exception:
                pass

            try:
                if final_url.lower().endswith(".pdf"):
                    return EXTERNAL_REDIRECT if not same_etld1(url, final_url) else HTTP_5XX_SAFE
            except Exception:
                pass

            try:
                idle_timeout = _ms_left() // 2
                if idle_timeout >= 500:
                    await page.wait_for_load_state("networkidle", timeout=idle_timeout)
            except Exception:
                pass   

            try:
                body_timeout = _ms_left()
                if body_timeout >= 500:
                    await page.wait_for_selector("body", timeout=body_timeout)
            except Exception:
                pass

            html = await page.content()

            if is_cloudflare_html(html):
                return CLOUDFLARE_BLOCKED
            try:
                import re
                m = re.search(r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\s*\d+\s*;\s*url=([^"\'> ]+)', html, re.I)
                if m:
                    refresh_url = m.group(1)
                    if not same_etld1(url, refresh_url) and not is_trusted_destination(refresh_url):
                        return EXTERNAL_REDIRECT
            except Exception:
                    pass

            return html

        except Exception as e:
            logging.warning(f"Playwright failed to fetch {url}: {e} (url={url})")
            return INACCESSIBLE
        finally:
            try:
                if page:
                    await page.close()
            finally:
                if context:
                    await context.close()

async def fetch_pdf_playwright(url, timeout=15000):
    page, context = await get_page()
    try:
        # Some servers open in a viewer; expect a download either way.
        async with page.expect_download() as dl_info:
            await page.goto(url, wait_until="networkidle", timeout=timeout)
        download = await dl_info.value
        path = await download.path()
        if path:
            return Path(path).read_bytes()
    except Exception as e:
        logging.info(f"[pdf] playwright download failed for {url}: {e}")
        return None
    finally:
        await context.close()

# --- HANDLE HTML ---
def _clean_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
        tag.decompose()
    return soup.get_text("\n", strip=True)

async def handle_html(url):
    static_html = fetch_static_html(url)

    if static_html == EXTERNAL_REDIRECT:
        return EXTERNAL_REDIRECT
    if static_html == HTTP_5XX_SAFE:
        return HTTP_5XX_SAFE
    if static_html == NOT_FOUND:
        logging.info(f"[404] static not found for {url}")
        return NOT_FOUND

    if not static_html:
        needs_dynamic = True
    else:
        if is_cloudflare_html(static_html):
            logging.info(f"[cloudflare] static challenge for {url}")
            dynamic_html = await fetch_dynamic_html(url)
            if dynamic_html == HTTP_5XX_SAFE:
                return HTTP_5XX_SAFE
            if dynamic_html in (CLOUDFLARE_BLOCKED, NOT_FOUND):
                logging.info(f"[cloudflare] dynamic challenge / 404 for {url}")
                return dynamic_html
            if dynamic_html and is_cloudflare_html(dynamic_html):
                logging.info(f"[cloudflare] dynamic still challenged for {url}")
                return CLOUDFLARE_BLOCKED
            if dynamic_html:
                    return preprocess_text(_clean_html(dynamic_html)) or INACCESSIBLE
            return CLOUDFLARE_BLOCKED

        spa_markers = (
            "ng-app", "ng-controller", "v-bind", "v-if", "react-root", 'id="__NEXT_DATA__"',
            "item.name", "sub.name",
            "Product.Ten", "Product.UnitPrice", "Product.TenDonViTinh", "Product.GiaKhuyenMai",
            "ComPany.Ten", "ComPany.DiaChiFull", "ComPany.Phone", "ComPany.Email",
        )
        low = static_html.lower()
        has_fw = any(f in low for f in ["vue", "react", "angular", "webpack", "require.js"])
        has_tpl = any(m in static_html for m in spa_markers)

        clean_text = _clean_html(static_html)
        clean_len = len(clean_text)

        needs_dynamic = has_fw or has_tpl or (clean_len < 500)

    if needs_dynamic:
        dynamic_html = await fetch_dynamic_html(url)
        if dynamic_html:
            if dynamic_html == EXTERNAL_REDIRECT:
                return EXTERNAL_REDIRECT
            if dynamic_html == HTTP_5XX_SAFE:
                return HTTP_5XX_SAFE
            if dynamic_html == NOT_FOUND:
                logging.info(f"[404] dynamic not found for {url}")
                return NOT_FOUND
            if dynamic_html == CLOUDFLARE_BLOCKED or is_cloudflare_html(dynamic_html):
                logging.info(f"[cloudflare] dynamic challenge for {url}")
                return CLOUDFLARE_BLOCKED

            return preprocess_text(_clean_html(dynamic_html)) or INACCESSIBLE

        return INACCESSIBLE

    return preprocess_text(clean_text) or INACCESSIBLE

# --- GET WEBSITE CONTEXT ---
async def get_website_context_async(url):
    ext = Path(urlparse(url).path).suffix.lower() or ".html"
    if ext == ".pdf":
        if not await is_accessible(url):
            logging.info(f"[context] not accessible: {url}")
            return INACCESSIBLE
        return await handle_pdf(url)
    return await handle_html(url)
    
# --- OVERLAP CHECK ---
def fmt_words(s):
    return ", ".join(map(str, sorted(s)))

def overlap_check(scraped_text, title_text):
    """
    Check how much overlap there is between two texts.
    If the overlap ratio is greater than 50%, treat them as similar.
    """
    words1 = set(scraped_text.lower().replace("_", " ").split())
    words2 = set(title_text.lower().replace("_", " ").split())
    overlap = words1.intersection(words2)
    non_overlap = words2 - words1
    ratio = len(overlap) / len(words2) if words2 else 0
    logging.info("Overlap ratio: %s", ratio)
    logging.info("Non-overlapped words: %s", fmt_words(non_overlap))
    logging.info("Title/Description text: %s", title_text[:500])
    logging.info("Scraped text: %s", scraped_text[:500])
    return (ratio > 0.5)

# --- CHECK SUBPAGE CONTEXT ---
async def check_subpage_context(
    domain: str,
    backlink: str,
    title_text: str,
    content_text: str,
    final_label: str,
    score: float,
    not_found_backlinks: Optional[Set[str]] = None
) -> "OutputEntry":
    redir = extract_redirect_target(backlink)
    if redir:
        src = f"https://{domain}"
        if not same_etld1(src, redir) and not is_trusted_destination(redir):
            logging.info(f"[redirector] external redirect param -> keep spam label ({backlink} -> {redir})")
            return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)
    try:
        subpage_context = await get_website_context_async(backlink)
    except Exception as e:
        logging.exception(f"[check_subpage] failed for {backlink}: {e}")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

    if subpage_context == EXTERNAL_REDIRECT:
        logging.info(f"[check_subpage] external 3xx redirect -> keep spam label")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

    if subpage_context == CLOUDFLARE_BLOCKED:
        redir = extract_redirect_target(backlink)
        if redir and not same_etld1(f"https://{domain}", redir) and not is_trusted_destination(redir):
            logging.info("[first-pass redirector] cloudflare on external redirect -> mark spam")
            return OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=score)
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)

    if subpage_context == NOT_FOUND:
        logging.info(f"[check_subpage] {backlink} 404 not found -> keep spam label")
        if not_found_backlinks is not None:
            not_found_backlinks.add(backlink)
        return OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=score)
    
    if subpage_context == HTTP_5XX_SAFE:
        logging.info(f"[check_subpage] {backlink} 5xx error -> mark safe")
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)

    if subpage_context == PDF_EMPTY_SENTINEL:
        logging.info(f"[check_subpage] {backlink} pdf accessible but empty -> mark safe")
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)

    if subpage_context in (INACCESSIBLE, PDF_FETCH_SENTINEL):
        logging.info(f"[check_subpage] {backlink} ctx={subpage_context!r} -> skip overlap")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

    logging.info(f"[check_subpage] running overlap for {backlink}")
    title_ok = overlap_check(subpage_context, title_text)
    content_ok = overlap_check(subpage_context, content_text)
    if title_ok and content_ok:
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)

    logging.info("404-tracked backlinks: %d", len(not_found_backlinks) if not_found_backlinks is not None else 0)
    return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

async def check_subpage_access_only(
    domain: str,
    backlink: str,
    final_label: str,
    score: float,
    not_found_backlinks: Optional[Set[str]] = None,
) -> "OutputEntry":
    """
    Second-pass helper: ONLY re-try accessibility/sentinels.
    Never runs overlap_check; if the page yields normal text, keep the original label.
    """
    redir = extract_redirect_target(backlink)
    if redir:
        src = f"https://{domain}"
        if not same_etld1(src, redir) and not is_trusted_destination(redir):
            logging.info("[second-pass redirector] external redirect param -> mark spam")
            return OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=score)
    try:
        subpage_context = await access_probe(backlink)
    except Exception as e:
        logging.exception(f"[second-pass access-only] failed for {backlink}: {e}")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

    if subpage_context == EXTERNAL_REDIRECT:
        logging.info("[second-pass access-only] external 3xx redirect -> mark spam")
        return OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=score)
    
    if subpage_context == CLOUDFLARE_BLOCKED:
        redir = extract_redirect_target(backlink)
        if redir and not same_etld1(f"https://{domain}", redir) and not is_trusted_destination(redir):
            logging.info("[second-pass redirector] cloudflare on external redirect -> mark spam")
            return OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=score)
        # trusted or internal redirect → safe
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)

    if subpage_context == NOT_FOUND:
        logging.info(f"[second-pass access-only] {backlink} 404 not found -> keep spam label")
        if not_found_backlinks is not None:
            not_found_backlinks.add(backlink)
        return OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=score)
    
    if subpage_context == HTTP_5XX_SAFE:
        logging.info(f"[second-pass access-only] {backlink} 5xx error -> mark safe")
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)

    if subpage_context == PDF_EMPTY_SENTINEL:
        logging.info(f"[second-pass access-only] {backlink} pdf accessible but empty -> mark safe")
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)

    if subpage_context in (INACCESSIBLE, PDF_FETCH_SENTINEL):
        logging.info(f"[second-pass access-only] {backlink} ctx={subpage_context!r} -> keep original")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

    logging.info(f"[second-pass access-only] {backlink} accessible text -> keep original label")
    return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

# --- CHECK TITLE THEN DESCRIPTION ---
async def check_title_then_desc(
    domain: str,
    backlink: str,
    title_text: str,
    desc_text: str,
    final_label: str,
    title_score: float,
    desc_score: float,
):
    """Title must pass overlap first; only then we check description.
       If title fails → return spam immediately."""
    try:
        subpage_context = await get_website_context_async(backlink)
    except Exception as e:
        logging.exception(f"[check_subpage] failed for {backlink}: {e}")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=title_score)

    if subpage_context == CLOUDFLARE_BLOCKED:
        logging.info(f"[check_subpage] {backlink} cloudflare -> mark safe")
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=title_score)

    if subpage_context == NOT_FOUND:
        logging.info(f"[check_subpage] {backlink} 404 not found -> keep spam")
        NOT_FOUND_BACKLINKS.add(backlink)
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=title_score)

    # PDF rules
    if subpage_context == PDF_EMPTY_SENTINEL:
        logging.info(f"[check_subpage] {backlink} pdf accessible but empty -> mark safe")
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=title_score)

    if subpage_context in (INACCESSIBLE, PDF_FETCH_SENTINEL):
        logging.info(f"[check_subpage] {backlink} ctx={subpage_context!r} -> skip overlap")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=title_score)

    # 1) Title must pass
    logging.info(f"[check_subpage] running title-overlap for {backlink}")
    if not overlap_check(subpage_context, title_text):
        logging.info(f"[check_subpage] title overlap insufficient -> keep spam")
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=title_score)

    # 2) Title passed → now check description
    if desc_text:
        logging.info(f"[check_subpage] running desc-overlap for {backlink}")
        if overlap_check(subpage_context, desc_text):
            return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=max(title_score, desc_score))
        else:
            logging.info(f"[check_subpage] title passed but description failed -> keep spam")

    # No description or it failed
    return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=title_score)

# --- PROCESS DOMAIN ---
async def process_domain(domain):
    if domain in domain_dict:
        return domain_dict[domain]
    ext = tldextract.extract(domain)
    normalized = f"{ext.domain}.{ext.suffix}"
    domain_dict[domain] = normalized
    return normalized