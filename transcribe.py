import os
import json
import static_ffmpeg

# Automatically link ffmpeg binaries
static_ffmpeg.add_paths()

import yt_dlp
from faster_whisper import WhisperModel
from tracker import PipelineTracker

def process_whisper_fallback():
    tracker = PipelineTracker()
    pending = tracker.get_pending()

    if not pending:
        print("No pending videos require Whisper processing.")
        return

    print(f"Loading Faster-Whisper 'small' Model on CPU (Optimized Speed)...")
    # Using 'small' model with int8 quantization for 4x faster execution
    model = WhisperModel("small", device="cpu", compute_type="int8")

    for idx, (video_id, title) in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] Processing: {title}")
        
        audio_path = f"data/audio/{video_id}.mp3"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128'  # Lower bitrate = faster processing
            }],
            'outtmpl': f"data/audio/{video_id}",
            'quiet': True
        }
        
        try:
            print("  -> Downloading audio...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            if not os.path.exists(audio_path):
                print(f"  -> Audio download failed for {video_id}")
                continue

            print("  -> Transcribing with Whisper (this may take 1-2 mins per video)...")
            segments, _ = model.transcribe(audio_path, language="hi", beam_size=1)
            
            transcript = []
            for s in segments:
                transcript.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})

            os.makedirs("data/transcripts", exist_ok=True)
            with open(f"data/transcripts/{video_id}.json", "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)

            os.remove(audio_path)
                
            tracker.update_status(video_id, "TRANSCRIBED", "WHISPER")
            print(f"  ✓ Saved transcript! ({len(transcript)} segments)")
            
        except Exception as e:
            print(f"  x Error processing video {video_id}: {e}")

if __name__ == "__main__":
    process_whisper_fallback()