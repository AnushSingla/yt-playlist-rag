import json
import os
import sqlite3
from dotenv import load_dotenv  # type: ignore
from fastembed import TextEmbedding  # type: ignore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

# Load environment variables from root .env file
load_dotenv()

COLLECTION_NAME = "dsa_transcripts"
TRANSCRIPTS_DIR = "data/transcripts"
DB_PATH = "data/pipeline_state.db"


class VectorStore:

    def __init__(self):
        self.collection_name = COLLECTION_NAME

        # 1. Initialize FastEmbed model (lightweight, no PyTorch dependency)
        # Model dimension for BAAI/bge-small-en-v1.5 is 384
        self.encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # 2. Check for Cloud Qdrant Environment Variables
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            # Production / Cloud setup
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Local fallback for local development
            self.qdrant = QdrantClient(path="data/qdrant_db")

    def create_collection_if_not_exists(self):
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def encode_text(self, text: str) -> list[float]:
        """Utility method to encode a single string into a vector list using FastEmbed."""
        return list(self.encoder.embed([text]))[0].tolist()

    def index_transcripts(self):
        # Reset existing collection before re-indexing
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)

        self.create_collection_if_not_exists()

        video_titles = {}
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            video_titles = dict(
                cursor.execute("SELECT video_id, title FROM videos").fetchall()
            )
            conn.close()

        points = []
        point_id = 1

        if not os.path.exists(TRANSCRIPTS_DIR):
            print(f"Directory '{TRANSCRIPTS_DIR}' not found.")
            return

        json_files = [
            f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".json")
        ]
        print(f"Indexing {len(json_files)} transcript files...")

        for f_name in tqdm(json_files):
            video_id = f_name.replace(".json", "")
            title = video_titles.get(video_id, "DSA Video")
            file_path = os.path.join(TRANSCRIPTS_DIR, f_name)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)

                if isinstance(transcript_data, dict):
                    transcript_data = (
                        transcript_data.get("transcript")
                        or transcript_data.get("segments")
                        or transcript_data.get("data")
                        or []
                    )

                if not isinstance(transcript_data, list):
                    continue

                CHUNK_SIZE = 3
                for i in range(0, len(transcript_data), CHUNK_SIZE):
                    group = transcript_data[i : i + CHUNK_SIZE]
                    chunk_text = " ".join(
                        [
                            seg.get("text", "").strip()
                            for seg in group
                            if isinstance(seg, dict)
                        ]
                    ).strip()

                    if not chunk_text:
                        continue

                    # Precise start and end integer timestamp calculation
                    raw_start = (
                        group[0].get("start", 0)
                        if isinstance(group[0], dict)
                        else 0
                    )
                    start_time = int(float(raw_start))

                    raw_end = (
                        group[-1].get("end", 0)
                        if isinstance(group[-1], dict)
                        else 0
                    )
                    end_time = int(float(raw_end))

                    # Vector generation with FastEmbed
                    vector = self.encode_text(chunk_text)

                    payload = {
                        "video_id": video_id,
                        "title": title,
                        "text": chunk_text,
                        "start": start_time,
                        "end": end_time,
                        "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={start_time}s",
                    }

                    points.append(
                        PointStruct(
                            id=point_id, vector=vector, payload=payload
                        )
                    )
                    point_id += 1

                    if len(points) >= 500:
                        self.qdrant.upsert(
                            collection_name=self.collection_name, points=points
                        )
                        points = []

            except Exception as e:
                print(f"\nError processing {f_name}: {e}")

        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name, points=points
            )

        print(f"\n✓ INDEXING COMPLETE! {point_id - 1} chunks stored successfully.")
        
        # Safely close Qdrant connection to avoid destructor warnings
        self.qdrant.close()


if __name__ == "__main__":
    store = VectorStore()
    store.index_transcripts()