import os
import json
import sqlite3
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "dsa_transcripts")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TRANSCRIPTS_DIR = "data/transcripts"
DB_PATH = os.getenv("DB_PATH", "data/pipeline_state.db")


class VectorStore:
    def __init__(self):
        self.collection_name = COLLECTION_NAME
        qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
        qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip()

        # Connect to Qdrant Cloud if credentials are present, else fallback to local storage
        if qdrant_url and qdrant_api_key:
            print(f"Connecting to Qdrant Cloud: {qdrant_url[:40]}...")
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            print("Connecting to local Qdrant database: data/qdrant_db...")
            self.qdrant = QdrantClient(path="data/qdrant_db")

        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def create_collection_if_not_exists(self):
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def index_transcripts(self):
        if self.qdrant.collection_exists(self.collection_name):
            print(f"Deleting existing collection '{self.collection_name}' for clean sync...")
            self.qdrant.delete_collection(self.collection_name)

        self.create_collection_if_not_exists()

        video_titles = {}
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            video_titles = dict(cursor.execute("SELECT video_id, title FROM videos").fetchall())
            conn.close()

        points = []
        point_id = 1

        if not os.path.exists(TRANSCRIPTS_DIR):
            print(f"Directory {TRANSCRIPTS_DIR} not found.")
            return

        json_files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".json")]
        print(f"Indexing {len(json_files)} transcript files using overlapping chunks...")

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

                if not isinstance(transcript_data, list) or len(transcript_data) == 0:
                    continue

                CHUNK_SIZE = 6
                STRIDE = 2
                num_segs = len(transcript_data)

                for i in range(0, max(1, num_segs), STRIDE):
                    group = transcript_data[i : min(i + CHUNK_SIZE, num_segs)]
                    if not group:
                        continue

                    chunk_text = " ".join(
                        [seg.get("text", "").strip() for seg in group if isinstance(seg, dict)]
                    ).strip()

                    if not chunk_text or len(chunk_text) < 15:
                        continue

                    # Exact start and end integer timestamp calculation
                    raw_start = group[0].get("start", 0) if isinstance(group[0], dict) else 0
                    start_time = int(float(raw_start))

                    raw_end = group[-1].get("end", 0) if isinstance(group[-1], dict) else 0
                    end_time = int(float(raw_end))

                    vector = self.encoder.encode(chunk_text).tolist()

                    payload = {
                        "video_id": video_id,
                        "title": title,
                        "text": chunk_text,
                        "start": start_time,
                        "end": end_time,
                        "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={start_time}s",
                    }

                    points.append(PointStruct(id=point_id, vector=vector, payload=payload))
                    point_id += 1

                    if len(points) >= 300:
                        self.qdrant.upsert(collection_name=self.collection_name, points=points)
                        points = []

                    if i + CHUNK_SIZE >= num_segs:
                        break

            except Exception as e:
                print(f"\nError processing {f_name}: {e}")

        if points:
            self.qdrant.upsert(collection_name=self.collection_name, points=points)

        print(f"\n[SUCCESS] INDEXING COMPLETE! {point_id - 1} overlapping chunks stored in collection '{self.collection_name}'.")


if __name__ == "__main__":
    store = VectorStore()
    store.index_transcripts()