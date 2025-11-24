"""
Embedder module for generating text embeddings using OpenRouter API.

This module provides the Embedder class that handles:
- OpenRouter API client initialization
- Batch embedding generation with retry logic
- Checkpoint mechanism for crash recovery
- Progress tracking with tqdm
"""

import os
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from .config import BATCH_SIZE, CHECKPOINT_INTERVAL


class Embedder:
    """
    Text embedder using OpenRouter API (Qwen3-Embedding-8B model).

    Features:
    - Batch processing with configurable batch size
    - Automatic retry with exponential backoff
    - Checkpoint support for long-running tasks
    - Progress tracking with tqdm
    """

    def __init__(self, device: str = None):
        """
        Initialize OpenRouter API client for embeddings.

        Args:
            device: Ignored (kept for compatibility with local model interface)
        """
        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Please set it with your OpenRouter API key."
            )

        print(f"[Embedder] Initializing OpenRouter API client")
        print(f"[Embedder] Model: qwen/qwen3-embedding-8b")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = "qwen/qwen3-embedding-8b"
        self.embedding_dim = 4096  # Qwen3-Embedding-8B dimension

        print(f"[Embedder] Client initialized successfully")

    def encode(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        checkpoint_enabled: bool = False,
        checkpoint_dir: Path = None,
        checkpoint_id: str = None,
        validation_metadata: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Encode texts to embeddings using OpenRouter API with automatic retry and checkpoint support.

        Args:
            texts: List of texts to encode
            batch_size: Number of texts to process per API call
            checkpoint_enabled: Enable checkpoint mechanism for crash recovery
            checkpoint_dir: Directory to save checkpoint files (required if checkpoint_enabled=True)
            checkpoint_id: Unique identifier for this encoding task (e.g., "review_embeddings_Amazon_Fashion")
            validation_metadata: Metadata for checkpoint validation (e.g., {'item_review_counts': [...]})

        Returns:
            numpy array of normalized embeddings, shape (len(texts), embedding_dim)
        """
        total_texts = len(texts)
        total_batches = (total_texts + batch_size - 1) // batch_size

        print(f"[Embedder] Encoding {total_texts} texts in {total_batches} batches (batch_size={batch_size})")

        # Checkpoint setup
        checkpoint_file = None
        if checkpoint_enabled:
            if not checkpoint_dir or not checkpoint_id:
                raise ValueError("checkpoint_dir and checkpoint_id required when checkpoint_enabled=True")
            checkpoint_file = Path(checkpoint_dir) / f"{checkpoint_id}_checkpoint.pkl"

        # Initialize or load from checkpoint
        all_embeddings = []
        start_batch = 0

        if checkpoint_file and checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                # Validate checkpoint
                if checkpoint_data.get('total_texts') != total_texts:
                    print(f"[WARN] Checkpoint text count mismatch: {checkpoint_data.get('total_texts')} != {total_texts}")
                    print(f"[WARN] Starting fresh encoding")
                elif checkpoint_data.get('model_name') != self.model_name:
                    print(f"[WARN] Checkpoint model mismatch: {checkpoint_data.get('model_name')} != {self.model_name}")
                    print(f"[WARN] Starting fresh encoding")
                elif validation_metadata:
                    # Validate caller-provided metadata
                    saved_metadata = checkpoint_data.get('validation_metadata', {})
                    mismatch = False

                    for key, expected_value in validation_metadata.items():
                        saved_value = saved_metadata.get(key)
                        if saved_value != expected_value:
                            print(f"[WARN] Checkpoint validation failed: {key} mismatch")
                            mismatch = True
                            break

                    if mismatch:
                        print(f"[WARN] Starting fresh encoding")
                    else:
                        # Checkpoint valid, resume
                        all_embeddings = checkpoint_data['embeddings']
                        start_batch = checkpoint_data['processed_batches']
                        print(f"[SUCCESS] Resuming from batch {start_batch}/{total_batches} ({len(all_embeddings)} embeddings loaded)")
                else:
                    # No validation metadata, accept checkpoint
                    all_embeddings = checkpoint_data['embeddings']
                    start_batch = checkpoint_data['processed_batches']
                    print(f"[SUCCESS] Resuming from batch {start_batch}/{total_batches} ({len(all_embeddings)} embeddings loaded)")

            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint: {e}")
                print(f"[WARN] Starting fresh encoding")
                all_embeddings = []
                start_batch = 0

        # Process in batches with progress bar
        for batch_idx in tqdm(range(start_batch, total_batches),
                              desc="Encoding batches",
                              initial=start_batch,
                              total=total_batches):

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]

            # Retry logic: 3 attempts with exponential backoff
            max_retries = 3
            retry_delays = [5, 15, 30]  # seconds

            for attempt in range(max_retries):
                try:
                    # Call OpenRouter API
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch_texts
                    )

                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]

                    if isinstance(all_embeddings, list):
                        all_embeddings.extend(batch_embeddings)
                    else:
                        all_embeddings = batch_embeddings

                    # Success, break retry loop
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        # Not the last attempt, retry
                        wait_time = retry_delays[attempt]
                        print(f"\n[RETRY] Batch {batch_idx + 1} failed (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"[RETRY] Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed, raise exception
                        print(f"\n[ERROR] Batch {batch_idx + 1} failed after {max_retries} attempts: {e}")
                        print(f"[ERROR] Failed at batch {batch_idx + 1}, total processed: {len(all_embeddings)} embeddings")
                        raise

            # Save checkpoint periodically
            if checkpoint_file and (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_data = {
                    'embeddings': all_embeddings,
                    'processed_batches': batch_idx + 1,
                    'total_batches': total_batches,
                    'total_texts': total_texts,
                    'batch_size': batch_size,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim,
                    'validation_metadata': validation_metadata,
                    'checkpoint_version': '1.0'
                }

                try:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    print(f"\n[Checkpoint] Saved at batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1)*100//total_batches}%)")
                except Exception as e:
                    print(f"\n[WARN] Failed to save checkpoint: {e}")

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Normalize embeddings (L2 normalization)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized_embeddings = embeddings_array / (norms + 1e-8)

        # Clean up checkpoint on success
        if checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"[SUCCESS] Checkpoint deleted (encoding completed)")
            except Exception as e:
                print(f"[WARN] Failed to delete checkpoint: {e}")

        return normalized_embeddings

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension (for compatibility with original code)"""
        return self.embedding_dim
