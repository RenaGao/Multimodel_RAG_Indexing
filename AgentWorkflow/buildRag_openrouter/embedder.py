"""
Embedder module for generating text embeddings using OpenRouter API.

This module provides the Embedder class that handles:
- OpenRouter API client initialization
- Batch embedding generation with retry logic
- Checkpoint mechanism for crash recovery with memmap support
- Progress tracking with tqdm
- Memory-efficient processing using memory-mapped files
"""

import os
import time
import pickle
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    - Memory-mapped file support for low-memory processing
    - Automatic migration from legacy checkpoint format
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
        checkpoint_dir: Optional[Path] = None,
        checkpoint_id: Optional[str] = None,
        validation_metadata: Optional[Dict[str, Any]] = None,
        use_memmap: bool = False,
        return_memmap: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings using OpenRouter API with automatic retry and checkpoint support.

        Args:
            texts: List of texts to encode
            batch_size: Number of texts to process per API call
            checkpoint_enabled: Enable checkpoint mechanism for crash recovery
            checkpoint_dir: Directory to save checkpoint files (required if checkpoint_enabled=True)
            checkpoint_id: Unique identifier for this encoding task (e.g., "review_embeddings")
            validation_metadata: Metadata for checkpoint validation (e.g., {'item_review_counts': [...]})
            use_memmap: Use memory-mapped file for low-memory processing
            return_memmap: Return memmap reference instead of loading into RAM (requires use_memmap=True)

        Returns:
            numpy array of normalized embeddings, shape (len(texts), embedding_dim)
            If return_memmap=True, returns np.memmap instead of regular array
        """
        total_texts = len(texts)
        total_batches = (total_texts + batch_size - 1) // batch_size

        print(f"[Embedder] Encoding {total_texts} texts in {total_batches} batches (batch_size={batch_size})")
        if use_memmap:
            print(f"[Embedder] Using memory-mapped file for efficient memory usage")

        # Checkpoint setup
        checkpoint_file = None
        memmap_file = None

        if checkpoint_enabled:
            if not checkpoint_dir or not checkpoint_id:
                raise ValueError("checkpoint_dir and checkpoint_id required when checkpoint_enabled=True")
            checkpoint_file = Path(checkpoint_dir) / f"{checkpoint_id}_checkpoint.pkl"
            if use_memmap:
                memmap_file = Path(checkpoint_dir) / f"{checkpoint_id}_embeddings.dat"

        # Initialize embeddings storage
        embeddings_storage = None
        start_batch = 0

        # Try to load checkpoint
        if checkpoint_file and checkpoint_file.exists():
            embeddings_storage, start_batch = self._load_checkpoint(
                checkpoint_file,
                memmap_file,
                total_texts,
                total_batches,
                validation_metadata,
                use_memmap
            )

        # Create new storage if not loaded from checkpoint
        if embeddings_storage is None:
            if use_memmap and checkpoint_enabled:
                print(f"[Memmap] Creating new memory-mapped file: {memmap_file}")
                embeddings_storage = np.memmap(
                    memmap_file,
                    dtype='float32',
                    mode='w+',
                    shape=(total_texts, self.embedding_dim)
                )
            else:
                embeddings_storage = []

        # Process in batches with progress bar
        for batch_idx in tqdm(range(start_batch, total_batches),
                              desc="Encoding batches",
                              initial=start_batch,
                              total=total_batches):

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]

            # Call API with retry
            batch_embeddings = self._call_api_with_retry(batch_texts, batch_idx)

            # Store embeddings
            if use_memmap:
                # Write directly to memmap (disk, not RAM)
                embeddings_storage[start_idx:end_idx] = batch_embeddings
                embeddings_storage.flush()  # Ensure written to disk
            else:
                # Traditional in-memory list
                if isinstance(embeddings_storage, list):
                    embeddings_storage.extend(batch_embeddings)
                else:
                    embeddings_storage = batch_embeddings

            # Save checkpoint periodically
            if checkpoint_file and (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(
                    checkpoint_file,
                    memmap_file,
                    embeddings_storage,
                    batch_idx + 1,
                    total_batches,
                    total_texts,
                    batch_size,
                    validation_metadata,
                    use_memmap
                )

        # Convert to final format
        if use_memmap and return_memmap:
            # Return memmap reference for streaming processing
            print(f"[Embedder] Returning memmap reference (data stays on disk)")
            return embeddings_storage
        else:
            # Load into memory and normalize
            print(f"[Embedder] Loading embeddings into memory and normalizing...")
            embeddings_array = np.array(embeddings_storage, dtype=np.float32)

            # Normalize embeddings (L2 normalization)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            normalized_embeddings = embeddings_array / (norms + 1e-8)

            # Clean up
            if use_memmap:
                del embeddings_storage
                if memmap_file and memmap_file.exists():
                    memmap_file.unlink()
                    print(f"[Memmap] Temporary file deleted")

            if checkpoint_file and checkpoint_file.exists():
                checkpoint_file.unlink()
                print(f"[Checkpoint] Deleted (encoding completed)")

            return normalized_embeddings

    def _load_checkpoint(
        self,
        checkpoint_file: Path,
        memmap_file: Optional[Path],
        total_texts: int,
        total_batches: int,
        validation_metadata: Optional[Dict],
        use_memmap: bool
    ):
        """Load checkpoint and return (embeddings_storage, start_batch) or (None, 0) if failed"""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Basic validation
            if checkpoint_data.get('total_texts') != total_texts:
                print(f"[WARN] Checkpoint text count mismatch: {checkpoint_data.get('total_texts')} != {total_texts}")
                print(f"[WARN] Starting fresh encoding")
                return None, 0

            if checkpoint_data.get('model_name') != self.model_name:
                print(f"[WARN] Checkpoint model mismatch: {checkpoint_data.get('model_name')} != {self.model_name}")
                print(f"[WARN] Starting fresh encoding")
                return None, 0

            # Validate metadata if provided
            if validation_metadata and not self._validate_metadata(
                checkpoint_data.get('validation_metadata', {}),
                validation_metadata
            ):
                print(f"[WARN] Validation metadata mismatch, starting fresh encoding")
                return None, 0

            # Detect checkpoint format
            checkpoint_format = self._detect_checkpoint_format(checkpoint_data)

            if checkpoint_format == 'legacy':
                # Legacy checkpoint with embedded embeddings
                return self._handle_legacy_checkpoint(
                    checkpoint_data,
                    memmap_file,
                    total_texts,
                    total_batches,
                    validation_metadata,
                    use_memmap
                )

            elif checkpoint_format == 'memmap':
                # Memmap checkpoint
                return self._handle_memmap_checkpoint(
                    checkpoint_data,
                    total_batches
                )

        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print(f"[WARN] Starting fresh encoding")
            return None, 0

    def _detect_checkpoint_format(self, checkpoint_data: dict) -> str:
        """Detect checkpoint format version"""
        if checkpoint_data.get('use_memmap'):
            return 'memmap'
        elif 'embeddings' in checkpoint_data:
            return 'legacy'
        else:
            raise ValueError("Unknown checkpoint format")

    def _handle_legacy_checkpoint(
        self,
        checkpoint_data: dict,
        memmap_file: Optional[Path],
        total_texts: int,
        total_batches: int,
        validation_metadata: Optional[Dict],
        use_memmap: bool
    ):
        """Handle legacy checkpoint format (with embedded embeddings)"""
        print(f"[MIGRATION] Detected legacy checkpoint format")

        if not use_memmap:
            # Just load embeddings normally
            embeddings = checkpoint_data['embeddings']
            start_batch = checkpoint_data['processed_batches']
            print(f"[SUCCESS] Resuming from batch {start_batch}/{total_batches} ({len(embeddings)} embeddings loaded)")
            return embeddings, start_batch

        # Migrate to memmap format
        print(f"[MIGRATION] Converting to memmap format for efficient memory usage...")

        embeddings_memmap = self._migrate_legacy_to_memmap(
            checkpoint_data,
            memmap_file,
            total_texts
        )

        start_batch = checkpoint_data['processed_batches']
        print(f"[SUCCESS] Migration complete. Resuming from batch {start_batch}/{total_batches}")

        # Save migrated checkpoint
        self._save_checkpoint(
            checkpoint_data=None,  # Will be created in method
            checkpoint_file=memmap_file.parent / f"{memmap_file.stem.replace('_embeddings', '')}_checkpoint.pkl",
            memmap_file=memmap_file,
            embeddings_storage=embeddings_memmap,
            processed_batches=start_batch,
            total_batches=total_batches,
            total_texts=total_texts,
            batch_size=checkpoint_data.get('batch_size', BATCH_SIZE),
            validation_metadata=validation_metadata,
            use_memmap=True
        )

        return embeddings_memmap, start_batch

    def _migrate_legacy_to_memmap(
        self,
        checkpoint_data: dict,
        memmap_file: Path,
        total_texts: int
    ) -> np.memmap:
        """Migrate legacy checkpoint embeddings to memmap file"""
        legacy_embeddings = checkpoint_data['embeddings']
        num_existing = len(legacy_embeddings)

        print(f"[MIGRATION] Extracting {num_existing:,} embeddings from checkpoint...")

        # Create memmap file
        embeddings_memmap = np.memmap(
            memmap_file,
            dtype='float32',
            mode='w+',
            shape=(total_texts, self.embedding_dim)
        )

        # Copy existing embeddings to memmap
        legacy_array = np.array(legacy_embeddings, dtype=np.float32)
        embeddings_memmap[:num_existing] = legacy_array
        embeddings_memmap.flush()

        file_size_gb = memmap_file.stat().st_size / (1024**3)
        print(f"[MIGRATION] Wrote {num_existing:,} embeddings to memmap file")
        print(f"[MIGRATION] Memmap file: {memmap_file}")
        print(f"[MIGRATION] File size: {file_size_gb:.2f} GB")

        # Free memory
        del legacy_embeddings
        del legacy_array
        gc.collect()

        return embeddings_memmap

    def _handle_memmap_checkpoint(self, checkpoint_data: dict, total_batches: int):
        """Handle memmap checkpoint format"""
        memmap_path = Path(checkpoint_data.get('memmap_file'))

        if not memmap_path.exists():
            print(f"[ERROR] Memmap file not found: {memmap_path}")
            print(f"[WARN] Cannot resume, starting fresh encoding")
            return None, 0

        # Load memmap
        embeddings_memmap = np.memmap(
            memmap_path,
            dtype='float32',
            mode='r+',
            shape=(checkpoint_data['total_texts'], self.embedding_dim)
        )

        start_batch = checkpoint_data['processed_batches']
        processed_embeddings = start_batch * checkpoint_data.get('batch_size', BATCH_SIZE)
        print(f"[SUCCESS] Resuming from batch {start_batch}/{total_batches} ({processed_embeddings:,} embeddings loaded)")

        return embeddings_memmap, start_batch

    def _save_checkpoint(
        self,
        checkpoint_file: Path,
        memmap_file: Optional[Path],
        embeddings_storage,
        processed_batches: int,
        total_batches: int,
        total_texts: int,
        batch_size: int,
        validation_metadata: Optional[Dict],
        use_memmap: bool,
        checkpoint_data: Optional[dict] = None
    ):
        """Save checkpoint to disk"""
        if checkpoint_data is None:
            checkpoint_data = {}

        if use_memmap:
            # Memmap format: only metadata, no embeddings
            checkpoint_data.update({
                'use_memmap': True,
                'memmap_file': str(memmap_file.absolute()),
                'processed_batches': processed_batches,
                'total_batches': total_batches,
                'total_texts': total_texts,
                'batch_size': batch_size,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'validation_metadata': validation_metadata,
                'checkpoint_version': '2.0'  # Memmap version
            })
        else:
            # Legacy format: embed all embeddings
            checkpoint_data.update({
                'embeddings': embeddings_storage,
                'processed_batches': processed_batches,
                'total_batches': total_batches,
                'total_texts': total_texts,
                'batch_size': batch_size,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'validation_metadata': validation_metadata,
                'checkpoint_version': '1.0'  # Legacy version
            })

        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            checkpoint_size_kb = checkpoint_file.stat().st_size / 1024

            if use_memmap:
                memmap_size_mb = memmap_file.stat().st_size / (1024**2)
                print(f"\n[Checkpoint] Saved at batch {processed_batches}/{total_batches} ({processed_batches*100//total_batches}%)")
                print(f"[Checkpoint] Metadata: {checkpoint_size_kb:.1f} KB, Data: {memmap_size_mb:.1f} MB")
            else:
                print(f"\n[Checkpoint] Saved at batch {processed_batches}/{total_batches} ({processed_batches*100//total_batches}%)")
                print(f"[Checkpoint] File size: {checkpoint_size_kb/1024:.1f} MB")

            gc.collect()  # Force garbage collection after checkpoint

        except Exception as e:
            print(f"\n[WARN] Failed to save checkpoint: {e}")

    def _validate_metadata(self, saved: dict, expected: dict) -> bool:
        """Validate checkpoint metadata matches expectations"""
        for key, expected_value in expected.items():
            saved_value = saved.get(key)
            if saved_value != expected_value:
                print(f"[WARN] Metadata mismatch for key '{key}'")
                return False
        return True

    def _call_api_with_retry(self, batch_texts: List[str], batch_idx: int) -> list:
        """Call OpenRouter API with automatic retry logic"""
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
                return [item.embedding for item in response.data]

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
                    raise

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension (for compatibility with original code)"""
        return self.embedding_dim
