import os
import glob
import whisper
import logging
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────

def batch_transcribe(
    input_dir="videos",
    output_dir="transcripts",
    model_name="medium"
):
    """
    Transcribe all media files in `input_dir` into .txt files in `output_dir`,
    logging progress and errors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    logger.info(f"Loading Whisper model '{model_name}'…")
    model = whisper.load_model(model_name)
    logger.info("Model loaded successfully.")

    # Gather all files
    exts = ("*.mp4", "*.mp3", "*.wav", "*.mov", "*.mkv")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    total = len(files)
    if total == 0:
        logger.warning(f"No media files found in '{input_dir}'.")
        return

    logger.info(f"Found {total} files. Beginning transcription…")

    # Process with progress bar
    for idx, vid in enumerate(tqdm(files, desc="Transcribing", unit="file"), 1):
        base = os.path.basename(vid)
        name, _ = os.path.splitext(base)
        out_txt = os.path.join(output_dir, f"{name}.txt")

        if os.path.exists(out_txt):
            logger.info(f"[{idx}/{total}] Skipping '{base}', transcript already exists.")
            continue

        logger.info(f"[{idx}/{total}] Transcribing '{base}' → '{out_txt}'")
        try:
            result = model.transcribe(vid)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(result["text"])
            logger.info(f"[{idx}/{total}] Saved '{out_txt}'")
        except Exception as e:
            logger.error(f"[{idx}/{total}] Failed to transcribe '{base}': {e}")

    logger.info("Batch transcription complete.")

if __name__ == "__main__":
    batch_transcribe()
