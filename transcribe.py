import os
import whisper

def transcribe_file(input_path: str, model_name: str = "medium", output_dir: str = "transcripts") -> str:
    """
    Transcribes the given audio/video file to text using OpenAI's Whisper with logging.
    Saves a .txt file in the `output_dir` and returns its path.
    """
    print(f"[Transcribe] Starting transcription for: {input_path}")
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Load Whisper model
    print(f"[Transcribe] Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    print(f"[Transcribe] Model loaded.")

    # Transcribe
    print(f"[Transcribe] Transcribing audio. This may take a while...")
    result = model.transcribe(input_path)
    print(f"[Transcribe] Transcription completed. Writing output...")

    # Build output filename
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    output_path = os.path.join(output_dir, f"{name}.txt")

    # Write transcript to disk
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"[Transcribe] Transcript saved to: {output_path}")

    return output_path
