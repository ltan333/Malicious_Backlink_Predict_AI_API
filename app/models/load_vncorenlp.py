# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import os
import py_vncorenlp

from app.cores.logging import logging

# --- MODEL ---
rdrsegmenter = None

def load_vncorenlp():
    global rdrsegmenter
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))          # -> app/models
        vncorenlp_dir = os.path.join(base_dir, "models_phobert", "vncorenlp")
        os.makedirs(vncorenlp_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")):
            logging.info("Downloading VnCoreNLP model...")
            py_vncorenlp.download_model(save_dir=vncorenlp_dir)
        rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_dir, annotators=["wseg"])
        logging.info("VnCoreNLP segmenter loaded")
    except Exception as e:
        logging.exception("Segmenter init failed; continuing without it: %s", e)