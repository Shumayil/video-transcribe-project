import os
import glob
import shutil
import subprocess
import logging
import torch
import whisper
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def split_audio(input_path: str, segment_dir: str, segment_length: int = 60):
    """
    Extracts audio from input video and splits into WAV segments of segment_length seconds.
    Saves segments as WAV files in segment_dir.
    """
    os.makedirs(segment_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "segment", "-segment_time", str(segment_length),
        "-reset_timestamps", "1",
        os.path.join(segment_dir, "seg_%04d.wav")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def batch_transcribe(
    input_dir: str = "videos",
    output_dir: str = "transcripts",
    model_name: str = "medium",
    segment_length: int = 60
) -> None:
    """
    Transcribes each video in input_dir into its own .txt in output_dir.
    Uses GPU (FP16) if available, else CPU (FP32), and splits audio to avoid memory errors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32" if device == "cuda" else "float32"
    logger.info(f"Loading Whisper model '{model_name}' on {device.upper()} ({compute_type})…")
    try:
        model = whisper.load_model(model_name, device=device, compute_type=compute_type)
    except TypeError:
        model = whisper.load_model(model_name, device=device)
    logger.info("Model loaded successfully.")

    # Gather video files
    extensions = ("*.mp4", "*.mov", "*.mkv", "*.mp3", "*.wav")
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(input_dir, ext)))
    total = len(videos)
    if total == 0:
        logger.warning(f"No video/audio files found in '{input_dir}'.")
        return
    logger.info(f"Found {total} files to transcribe in '{input_dir}'.")

    # Process each video
    for idx, vid in enumerate(videos, start=1):
        base = os.path.basename(vid)
        name, _ = os.path.splitext(base)
        out_txt = os.path.join(output_dir, f"{name}.txt")

        logger.info(f"[{idx}/{total}] Processing '{base}'…")
        if os.path.exists(out_txt):
            logger.info(f"[{idx}/{total}] Skipping '{base}', transcript exists.")
            continue

        # Split audio to segments
        seg_dir = os.path.join("tmp_segments", name)
        logger.info(f"[{idx}/{total}] Splitting audio into {segment_length}s segments…")
        try:
            split_audio(vid, seg_dir, segment_length)
        except Exception as e:
            logger.error(f"[{idx}/{total}] Failed splitting audio: {e}")
            continue

        # Transcribe each segment
        texts = []
        seg_files = sorted(glob.glob(os.path.join(seg_dir, "seg_*.wav")))
        for seg in tqdm(seg_files, desc=f"Transcribing {name}", unit="seg"):
            try:
                result = model.transcribe(seg)
                texts.append(result.get("text", ""))
            except Exception as e:
                logger.error(f"Error on segment {seg}: {e}")

        # Write full transcript
        try:
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            logger.info(f"[{idx}/{total}] Saved transcript → '{out_txt}'")
        except Exception as e:
            logger.error(f"[{idx}/{total}] Failed writing transcript: {e}")

        # Cleanup segments
        shutil.rmtree(seg_dir, ignore_errors=True)

    logger.info("Batch transcription complete.")

if __name__ == "__main__":
    batch_transcribe()
