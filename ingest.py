import os
import sys
import yt_dlp # type: ignore
from tracker import PipelineTracker

def fetch_playlist_metadata(playlist_url: str, tracker: PipelineTracker):
    print("-> Fetching playlist metadata...")
    ydl_opts = {'extract_flat': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        for entry in info.get('entries', []):
            tracker.add_video(entry['id'], entry['title'], entry.get('duration', 0))

def download_native_captions(video_id: str, output_dir="data/captions") -> bool:
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'skip_download': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['hi', 'en'],
        'subtitlesformat': 'vtt',
        'outtmpl': f'{output_dir}/{video_id}',
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        for file in os.listdir(output_dir):
            if file.startswith(video_id) and file.endswith('.vtt'):
                return True
    except Exception:
        pass
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <PLAYLIST_URL>")
        sys.exit(1)

    playlist_url = sys.argv[1]
    tracker = PipelineTracker()
    
    fetch_playlist_metadata(playlist_url, tracker)
    pending = tracker.get_pending()
    print(f"Logged {len(pending)} videos into the database.")

    print("\n-> Attempting native subtitle downloads...")
    for video_id, title in pending:
        if download_native_captions(video_id):
            print(f"  [CAPTIONS FOUND] {title}")
            tracker.update_status(video_id, "CAPTIONED", "YT_CAPTION")
        else:
            print(f"  [NO CAPTIONS] {title} (marked for Whisper)")