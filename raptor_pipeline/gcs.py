import os
from google.cloud import storage


class GCSCheckpointStore:
    def __init__(self, bucket_name: str, prefix: str, local_dir: str):
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.local_dir = local_dir

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def pull(self):
        blobs = list(self.bucket.list_blobs(prefix=self.prefix))
        if not blobs:
            print("ℹ️ No checkpoints in GCS (first run)")
            return

        for blob in blobs:
            rel_path = blob.name.replace(self.prefix, "")
            local_path = os.path.join(self.local_dir, rel_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)

        print("✅ Checkpoints downloaded")

    def push(self):
        print("⬆️ Uploading checkpoints to GCS...")
        for root, _, files in os.walk(self.local_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.local_dir)

                blob = self.bucket.blob(f"{self.prefix}{rel_path}")
                blob.upload_from_filename(full_path)

        print("✅ Checkpoints uploaded")
