# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import tldextract
from urllib.parse import urlparse, parse_qs, unquote

from app.cores import config
from app.cores.logging import logging

# --- REDIRECT CHECK ---
def etld1(host: str) -> str:
    """Extract effective top-level domain + 1 (e.g., 'example.com' from 'sub.example.com')"""
    e = tldextract.extract(host or "")
    return f"{e.domain}.{e.suffix}" if e.suffix else host or ""

def same_etld1(a_url: str, b_url: str) -> bool:
    """Check if two URLs belong to the same effective top-level domain"""
    return etld1(urlparse(a_url).hostname or "") == etld1(urlparse(b_url).hostname or "")

def etld1_of(host: str) -> str:
    """Get the effective top-level domain of a host"""
    e = tldextract.extract(host or "")
    return f"{e.domain}.{e.suffix}" if e.suffix else (host or "")

def is_trusted_destination(dest_url: str) -> bool:
    """Check if a destination URL is from a trusted domain"""
    host = urlparse(dest_url).hostname or ""
    host_etld1 = etld1_of(host)
    if not host_etld1:
        return False
    if host_etld1 in config.TRUSTED_REDIRECT_DOMAINS:
        return True
    return any(host_etld1.endswith(suf) for suf in config.TRUSTED_REDIRECT_SUFFIXES)

def extract_redirect_target(backlink: str) -> str | None:
    """Extract redirect target from URL parameters"""
    try:
        p = urlparse(backlink)
        qs = parse_qs(p.query)
        for k in config.REDIRECT_PARAM_KEYS:
            if k in qs and qs[k]:
                cand = unquote(qs[k][0])
                u = urlparse(cand)
                if u.scheme in ("http","https") and u.netloc:
                    return cand
    except Exception:
        pass
    return None
