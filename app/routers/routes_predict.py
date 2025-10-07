# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import asyncio
from typing import List
from fastapi import Depends, HTTPException, APIRouter

from app.cores import config
from app.schemas.schemas import InputEntry, OutputEntry
from app.cores.security import verify_access_token
import app.models.load_models as lm
from app.utils.preprocessing import preprocess_text, INACCESSIBLE, CLOUDFLARE_BLOCKED, NOT_FOUND, HTTP_5XX_SAFE, EXTERNAL_REDIRECT
from app.services.scraping_service import process_domain, get_website_context_async, check_title_then_desc, check_subpage_context, check_subpage_access_only, NOT_FOUND_BACKLINKS
from app.services.redirect_service import extract_redirect_target, same_etld1, is_trusted_destination
from app.services.cache_service import save_cache, homepage_cache
from app.cores.logging import logging

router = APIRouter(prefix="", tags=["predict"])

# --- PREDICT ENDPOINT ---
# Endpoint to predict
@router.post("/predict", response_model=List[OutputEntry], dependencies=[Depends(verify_access_token)])
async def predict(input_data: List[InputEntry]):
    if lm.model is None or lm.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    MAX_BATCH_SIZE = config.MAX_BATCH_SIZE
    if len(input_data) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch size exceeds (max={MAX_BATCH_SIZE})")

    # --- STEP 1: Prepare domains and tasks concurrently ---
    unique_domains = await asyncio.gather(*(process_domain(entry.domain) for entry in input_data))
    unique_domains_set = set(unique_domains)

    logging.info("[home] normalized domains: %s", unique_domains_set)
    for d in unique_domains_set:
        logging.info("[home] cache %s: %s", "HIT" if d in homepage_cache else "MISS", d)

    homepage_tasks = {
        domain: asyncio.create_task(get_website_context_async(f"https://{domain}/"))
        for domain in unique_domains_set if domain not in homepage_cache
    }

    # Start classification tasks for titles and contents concurrently
    titles_all = [preprocess_text(entry.title) for entry in input_data]
    contents_all = [f"{titles_all[i]} {preprocess_text(entry.description)}" for i, entry in enumerate(input_data)]

    title_classify_task = asyncio.create_task(lm.classify_batch(titles_all))
    content_classify_task = asyncio.create_task(lm.classify_batch(contents_all))

    # Await homepage fetch concurrently
    homepage_results = await asyncio.gather(*homepage_tasks.values(), return_exceptions=True)
    homepage_context_map = dict(zip(homepage_tasks.keys(), homepage_results))

    # --- STEP 2: Classify and cache accessible homepages in one batch ---
    homepages_to_classify = [
        (domain, context)
        for domain, context in homepage_context_map.items()
        if (not isinstance(context, Exception)) and (context not in (INACCESSIBLE, CLOUDFLARE_BLOCKED, NOT_FOUND, HTTP_5XX_SAFE, EXTERNAL_REDIRECT))
    ]

    updated_homepages = False

    if homepages_to_classify:
        homepage_labels = await lm.classify_batch([preprocess_text(ctx) for _, ctx in homepages_to_classify])
        for (domain, _), (label, _) in zip(homepages_to_classify, homepage_labels):
            homepage_cache[domain] = label
            updated_homepages = True

    if updated_homepages:
        await save_cache()

    # --- STEP 3: Finish title and content classification ---
    title_results, content_results = await asyncio.gather(title_classify_task, content_classify_task)
    results = [None] * len(input_data)
    skipped_safe_candidates = []
    not_found_backlinks: set[str] = set()

    for i, (entry, domain) in enumerate(zip(input_data, unique_domains)):
        homepage_label = homepage_cache.get(domain)
        if homepage_label == "gambling":
            logging.info(f"[shortcut] homepage gambling for {domain} -> skip overlap")
            results[i] = OutputEntry(domain=domain, backlink=entry.backlink, label="Cờ bạc", score=1.0)

    # --- STEP 4: Use classification results for remaining entries ---    
    subpage_promises = []
    label_map = {
        "gambling": "Cờ bạc",
        "movies": "Phim lậu"
        # "ecommerce": "Cờ bạc",
        # "services": "Cờ bạc"
    }

    for i, (entry, domain) in enumerate(zip(input_data, unique_domains)):
        if results[i] is not None:
            continue

        title_label, title_score = title_results[i]
        content_label, content_score = content_results[i]

        if title_label in label_map:
            label_text = label_map[title_label]
            subpage_promises.append((
                i,
                asyncio.create_task(
                    check_subpage_context(
                        domain, entry.backlink,
                        titles_all[i],
                        contents_all[i],
                        label_text, title_score,
                        not_found_backlinks
                    )
                )
            ))
            continue

        if content_label in label_map:
            label_text = label_map[content_label]
            subpage_promises.append((
                i,
                asyncio.create_task(
                    check_subpage_context(
                        domain, entry.backlink,
                        titles_all[i],
                        contents_all[i],
                        label_text, content_score,
                        not_found_backlinks
                    )
                )
            ))
            continue

        redir = extract_redirect_target(entry.backlink)
        if redir and not same_etld1(f"https://{domain}", redir) and not is_trusted_destination(redir):
            logging.info("[redirector] external redirect param (first pass) -> mark spam")
            results[i] = OutputEntry(domain=domain, backlink=entry.backlink, label="Cờ bạc", score=content_score)
            continue

        results[i] = OutputEntry(domain=domain, backlink=entry.backlink, label="An toàn", score=content_score)
        skipped_safe_candidates.append((i, domain, entry.backlink, contents_all[i], "An toàn", content_score))

    if subpage_promises:
        for idx, task in subpage_promises:
            try:
                res = await task
            except Exception:
                res = None
            if isinstance(res, OutputEntry):
                results[idx] = res

    for i, res in enumerate(results):
        if res is None:
            entry = input_data[i]
            domain = unique_domains[i]
            results[i] = OutputEntry(domain=domain, backlink=entry.backlink, label="Predict error", score=0.0)

    # --- STEP 5: Adjust labels for low spam ratio ---
    spam_count = sum(1 for r in results if r.label in config.SPAM_LABELS)

    logging.info("spam count (pre-dampen): %s", spam_count)
    logging.info("length of input_data: %s", len(input_data))
    logging.info("spam ratio (pre-dampen): %s", spam_count / len(input_data))

    if spam_count / len(input_data) <= 0.15:
        for r in results:
            # if r.backlink in not_found_backlinks:
            #     logging.info("STEP5: skip dampen (404) %s", r.backlink)
            #     continue
            domain = await process_domain(r.domain)
            homepage_label = homepage_cache.get(domain)
            if homepage_label != "gambling" and r.label in config.SPAM_LABELS:
                r.label = "An toàn"

    # Recompute after dampening
    spam_count = sum(1 for r in results if r.label in config.SPAM_LABELS)
    logging.info("spam count (post-dampen): %s", spam_count)
    logging.info("spam ratio (post-dampen): %s", spam_count / len(input_data) if len(input_data) else 0)

    # --- STEP 6: Second pass for skipped backlinks if spam remains ---
    final_spam_count = sum(1 for r in results if r.label in config.SPAM_LABELS)
    if final_spam_count > 0 and skipped_safe_candidates:
        second_pass_tasks = [
            asyncio.create_task(
                check_subpage_access_only(domain, backlink, final_label, score, not_found_backlinks)
            )
            for (idx, domain, backlink, text, final_label, score) in skipped_safe_candidates
        ]
        second_pass_results = await asyncio.gather(*second_pass_tasks, return_exceptions=True)

        for ((idx, _, _, _, _, _), out) in zip(skipped_safe_candidates, second_pass_results):
            if isinstance(out, Exception):
                logging.exception("second-pass subpage failed idx=%s", idx)
                continue
            prev = results[idx].label
            results[idx] = out
            logging.info("%s -> %s (url=%s)", prev, out.label, out.backlink)

    # --- FINAL: log spam count after all passes/tweaks ---
    final_spam_count = sum(1 for r in results if r.label in config.SPAM_LABELS)
    total = len(results)
    logging.info("FINAL spam count: %d / %d (ratio=%.6f)",
                final_spam_count, total, (final_spam_count / total if total else 0.0))
    redirector_spam = sum(1 for r in results if "Redirect.aspx" in (r.backlink or "") and r.label in config.SPAM_LABELS)
    not_found_spam  = sum(1 for r in results if (r.backlink in not_found_backlinks) and r.label in config.SPAM_LABELS)
    logging.info("redirector_spam=%d, not_found_spam=%d", redirector_spam, not_found_spam)

    return results