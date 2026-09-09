import os
import sqlite3
import static_ffmpeg
import yt_dlp

# Automatically link ffmpeg binaries
static_ffmpeg.add_paths()

os.makedirs("data/audio_downloads", exist_ok=True)

conn = sqlite3.connect("data/pipeline_state.db")
cursor = conn.cursor()
videos = cursor.execute("SELECT video_id, title FROM videos WHERE status = 'PENDING'").fetchall()

print(f"Downloading audio streams for {len(videos)} videos locally...")

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'data/audio_downloads/%(id)s.%(ext)s',
    'quiet': False,
    'ignoreerrors': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for idx, (vid, title) in enumerate(videos, 1):
        # Check if already downloaded (webm, m4a, or mp3)
        existing = [f for f in os.listdir("data/audio_downloads") if f.startswith(vid)]
        if existing:
            continue
            
        print(f"\n[{idx}/{len(videos)}] Fetching: {title[:40]}")
        ydl.download([f"https://www.youtube.com/watch?v={vid}"])

print("\n✓ Audio downloads complete!")