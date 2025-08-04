# api/routes.py
from fastapi import FastAPI, Depends, HTTPException, status
from typing import List
import asyncio
import httpx
from utils.logging import logging
from auth import verify_access_token, create_access_token
from schemas.schemas import Token, SecretKeyInput, InputEntry, OutputEntry
from config import const as config
from utils.cache import homepage_cache, save_cache
from utils.model import classify_batch, model, tokenizer
from utils.text_utils import clean_text, overlap_check, process_domain
from utils.scraper import get_website_context_async
from datetime import datetime, timezone, timedelta

# Global spam_count for this module
spam_count = 0

async def check_subpage_context(domain, backlink, text, final_label, score):
    """
    Fetch the backlink page and compare its content with the predicted text.
    If content is similar, mark as 'safe', else keep original label.
    """
    global spam_count
    subpage_context = clean_text(await get_website_context_async(backlink))
    if subpage_context == "inaccessible":
        spam_count += 1
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)
    
    if overlap_check(subpage_context, text):
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)
    
    spam_count += 1
    return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

# Endpoint to get access token
async def login_for_access_token(secret_input: SecretKeyInput):
    """
    Login endpoint to obtain an access token.
    Checks if the provided API key matches the server's key.
    """
    if secret_input.api_key != config.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    access_token_expires = timedelta(seconds=config.ACCESS_TOKEN_EXPIRE_SECOND)
    access_token = await create_access_token(
        data={"sub": "authenticated_user"}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint to get token
async def get_token_info(current_user: str = Depends(verify_access_token)):
    """
    Verifies the provided token and returns user information and expiration time.
    """
    return {"message": "Token is valid", "exp": datetime.fromtimestamp(current_user.get("exp")), "user": current_user.get("sub")}

# Endpoint to predict
async def predict(input_data: List[InputEntry]):
    """
    Main prediction endpoint.
    Steps:
    - Validate input and model
    - Perform classification on title and content in parallel
    - Detect suspicious categories (gambling, movies, ecommerce)
    - Use homepage/subpage scraping to refine decision
    - Cache homepage results for reuse
    - Return classified label and confidence score per input
    """
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer not loaded. Cannot process prediction.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    MAX_BATCH_SIZE = config.MAX_BATCH_SIZE
    if len(input_data) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch size exceeds (max={MAX_BATCH_SIZE})")

    titles = [clean_text(entry.title) for entry in input_data]
    descriptions = [clean_text(entry.description) for entry in input_data]
    contents = [f"{title} {description}".strip() for title, description in zip(titles, descriptions)]

    # Parallel batch classification
    title_task = asyncio.create_task(classify_batch(titles))
    content_task = asyncio.create_task(classify_batch(contents))
    title_results, content_results = await asyncio.gather(title_task, content_task)

    global spam_count
    spam_count = 0
    results = []
    subpage_promises = []
    subpage_mapping = []
    homepage_promises = {}
    homepage_domains_needed = set()
    pending_homepage_entries = []
    save_cache_required = False

    async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
        normalized_domains = await asyncio.gather(*(process_domain(entry.domain) for entry in input_data))
        for i, (entry, domain) in enumerate(zip(input_data, normalized_domains)):
            try:
                backlink = entry.backlink
                title_label, title_score = title_results[i]
                content_label, content_score = content_results[i]

                # Classify first 15 characters of title
                if title_label == "gambling":
                    subpage_promises.append(check_subpage_context(domain, backlink, titles[i], "Cờ bạc", title_score))
                    subpage_mapping.append((i, domain, backlink))
                    continue
                if title_label == "movies":
                    subpage_promises.append(check_subpage_context(domain, backlink, titles[i], "Phim lậu", title_score))
                    subpage_mapping.append((i, domain, backlink))
                    continue

                # Classify content
                if content_label == "gambling":
                    subpage_promises.append(check_subpage_context(domain, backlink, contents[i], "Cờ bạc", content_score))
                    subpage_mapping.append((i, domain, backlink))
                    continue
                if content_label == "movies":
                    subpage_promises.append(check_subpage_context(domain, backlink, contents[i], "Phim lậu", content_score))
                    subpage_mapping.append((i, domain, backlink))
                    continue

                # Mark safe if not ecommerce
                if content_label != "ecommerce":
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=content_score))
                    continue

                # Homepage check logic
                if domain in homepage_cache:
                    homepage_label = homepage_cache[domain]
                else:
                    homepage_domains_needed.add(domain)
                    pending_homepage_entries.append((i, domain, entry.backlink, content_score))
                    continue  # delay decision until homepage content is available

                # Decision using cached homepage label
                if homepage_label not in ["education", "government"]:
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=content_score))
                else:
                    subpage_promises.append(check_subpage_context(domain, backlink, contents[i], "Quảng cáo bán hàng", content_score))
                    subpage_mapping.append((i, domain, backlink))

            # Exception for each entry
            except Exception as e:
                logging.error(f"Error processing entry {getattr(entry, 'domain', None)}: {e}")
                results.append(OutputEntry(
                    domain=getattr(entry, 'domain', ''),
                    backlink=getattr(entry, 'backlink', ''),
                    label="Predict error",
                    score=0.0
                ))

    # Fire all homepage fetches in parallel
    homepage_tasks = {
        domain: asyncio.create_task(get_website_context_async(f"https://{domain}/"))
        for domain in homepage_domains_needed
    }
    homepage_results = await asyncio.gather(*homepage_tasks.values(), return_exceptions=True)
    homepage_context_map = dict(zip(homepage_tasks.keys(), homepage_results))

    # Classify homepage and resolve deferred decisions
    for i, domain, backlink, content_score in pending_homepage_entries:
        homepage_context = homepage_context_map.get(domain)

        if isinstance(homepage_context, Exception) or homepage_context == "inaccessible":
            results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=content_score))
            continue

        try:
            homepage_result = await classify_batch([homepage_context])
            homepage_label, homepage_score = homepage_result[0]
            homepage_cache[domain] = homepage_label
            save_cache_required = True

            if homepage_label not in ["education", "government"]:
                results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=content_score))
            else:
                subpage_promises.append(check_subpage_context(domain, backlink, contents[i], "Quảng cáo bán hàng", content_score))
                subpage_mapping.append((i, domain, backlink))

        except Exception as e:
            logging.error(f"Error classifying homepage for domain={domain}: {e}")
            results.append(OutputEntry(domain=domain, backlink=backlink, label="Predict error", score=0.0))

    # Resolve all subpage checks
    if subpage_promises:
        subpage_results = await asyncio.gather(*subpage_promises, return_exceptions=True)
        for result in subpage_results:
            if isinstance(result, OutputEntry):
                results.append(result)
            else:
                logging.warning(f"Subpage check failed: {result}")

    # Save homepage cache once
    if save_cache_required:
        await save_cache()

    if spam_count / len(input_data) <= 0.15:
        for r in results:
            if r.label in ["Cờ bạc", "Phim lậu", "Quảng cáo bán hàng"]:
                r.label = "An toàn"

    return results