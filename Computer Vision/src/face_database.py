"""
This module implements the storage layer described in PRIVACY_POLICY.md.
It stores faculty face embeddings in an encrypted file on the local filesystem,
using Fernet symmetric encryption from the `cryptography` library.

Key design decisions (all from the privacy policy):
  - Only embeddings are stored, never raw images.
  - The encrypted database file and the encryption key are stored in
    SEPARATE files, so stealing one without the other is useless.
  - The key file is set to mode 600 (owner-read/write only).
  - All data stays on-device so no cloud, no network.
  - Faculty can be added or removed individually without rebuilding
    the entire database.

How Fernet encryption works (simplified):
  1. You generate a random 32-byte key (once, stored in a key file).
  2. To encrypt: Fernet takes your plaintext bytes + the key, and produces
     a blob of ciphertext that looks like random garbage.
  3. To decrypt: Fernet takes the ciphertext + the same key, and gives you
     back the original plaintext bytes.
  4. Fernet also includes a tamper-detection mechanism (HMAC-SHA256), so
     if anyone modifies the encrypted file, decryption will fail with an
     InvalidToken error instead of silently producing garbage.

Under the hood, Fernet uses AES-128-CBC for encryption and HMAC-SHA256
for authentication. These are industry-standard algorithms trusted by
banks, governments, and security researchers worldwide.

Satisfies requirements 4.2 (local operation) and 4.3 (secure on-device storage) for M2.
"""

import json
import os
import stat
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cryptography.fernet import Fernet, InvalidToken

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "face_embeddings.enc"
_DEFAULT_KEY_PATH = _PROJECT_ROOT / "data" / ".encryption_key"

############################ Key management ###########################

def _generate_key(key_path: Path) -> bytes:
    """Generate a new Fernet encryption key and save it to disk.

    The key file is set to mode 600 (owner read/write only) so that
    other users on the system cannot read it. This is a basic but important security measure.

    Args:
        key_path: Where to save the key file.

    Returns:
        The generated key as bytes.
    """
    key = Fernet.generate_key()

    # Create parent directories if needed.
    key_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the key.
    key_path.write_bytes(key)

    # Set file permissions to 600 (owner read/write only).
    # This means only the user account running Knightro can read the key.
    # On Windows this is a no-op (Windows uses ACLs, not POSIX permissions),
    # but on macOS and Linux (including the Pi) it works correctly.
    try:
        os.chmod(str(key_path), stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        # If chmod fails (e.g., on some Windows configs), warn but continue.
        print(f"[face_database] WARNING: Could not set permissions on {key_path}")
        print(f"[face_database] Make sure this file is not world-readable.")

    print(f"[face_database] New encryption key generated at {key_path}")
    return key


def _load_key(key_path: Path) -> bytes:
    """Load the encryption key from disk, generating one if it doesn't exist.

    Args:
        key_path: Path to the key file.

    Returns:
        The encryption key as bytes.
    """
    if not key_path.exists():
        print(f"[face_database] No encryption key found. Generating a new one...")
        return _generate_key(key_path)

    return key_path.read_bytes()

def _serialize(data: Dict[str, List[np.ndarray]]) -> bytes:
    """Convert the embeddings dict to JSON bytes.

    Args:
        data: Dict mapping faculty names to lists of numpy embedding arrays.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    json_data = {
        name: [emb.tolist() for emb in embeddings]
        for name, embeddings in data.items()
    }
    return json.dumps(json_data, indent=2).encode("utf-8")


def _deserialize(raw: bytes) -> Dict[str, List[np.ndarray]]:
    """Convert JSON bytes back to an embeddings dict.

    Args:
        raw: UTF-8 encoded JSON bytes.

    Returns:
        Dict mapping faculty names to lists of numpy embedding arrays.
    """
    json_data = json.loads(raw.decode("utf-8"))
    result = {}
    for name, embeddings in json_data.items():
        # Handle both old format (single embedding) and new format (list of embeddings)
        if isinstance(embeddings[0], (int, float)):
            # Old format: single flat list -> wrap in a list
            result[name] = [np.array(embeddings, dtype=np.float32)]
        else:
            # New format: list of lists
            result[name] = [np.array(emb, dtype=np.float32) for emb in embeddings]
    return result

############################ The database class ###########################

class FaceDatabase:
    """Encrypted on-device database for faculty face embeddings.

    Usage:
        db = FaceDatabase()

        # Add a faculty member
        db.add_face("Dr. Smith", embedding_array)
        db.save()

        # Look up all enrolled faculty
        for name, embedding in db.get_all().items():
            print(f"{name}: {len(embedding)}-dim embedding")

        # Remove a faculty member
        db.remove_face("Dr. Smith")
        db.save()

    The database auto-loads from the encrypted file on creation (if the
    file exists). You must call save() explicitly after modifications —
    this is intentional so that multiple add/remove operations can be
    batched into a single write.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        key_path: Optional[Path] = None,
    ):
        """Create or load an encrypted face database.

        Args:
            db_path: Path to the encrypted database file.
                     Defaults to data/face_embeddings.enc
            key_path: Path to the encryption key file.
                      Defaults to data/.encryption_key
        """
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._key_path = key_path or _DEFAULT_KEY_PATH

        # Load the encryption key (generates one if first run).
        self._key = _load_key(self._key_path)
        self._fernet = Fernet(self._key)

        # Load existing data from disk, or start with an empty dict.
        self._data: Dict[str, List[np.ndarray]] = {}
        if self._db_path.exists():
            self._load()

    def _load(self) -> None:
        """Load and decrypt the database from disk."""
        try:
            encrypted_bytes = self._db_path.read_bytes()
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            self._data = _deserialize(decrypted_bytes)
            print(f"[face_database] Loaded {len(self._data)} enrolled faculty "
                  f"from {self._db_path}")
        except InvalidToken:
            print(f"[face_database] ERROR: Could not decrypt {self._db_path}")
            print(f"[face_database] The encryption key may have changed or "
                  f"the file may be corrupted.")
            print(f"[face_database] Starting with an empty database.")
            self._data = {}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[face_database] ERROR: Database file is corrupted: {e}")
            print(f"[face_database] Starting with an empty database.")
            self._data = {}

    def save(self) -> None:
        """Encrypt and write the database to disk.

        Call this after any add_face() or remove_face() operations.
        The entire database is re-encrypted and written atomically.
        """
        # Create the directory if it doesn't exist.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to JSON, then encrypt.
        plaintext_bytes = _serialize(self._data)
        encrypted_bytes = self._fernet.encrypt(plaintext_bytes)

        # Write to disk. We write to a temp file first, then rename,
        # so that a crash mid-write doesn't corrupt the database.
        # (This is called "atomic write" and is a standard practice
        # for any file that must not be left in a half-written state.)
        temp_path = self._db_path.with_suffix(".tmp")
        temp_path.write_bytes(encrypted_bytes)
        temp_path.rename(self._db_path)

        # Set file permissions to 600.
        try:
            os.chmod(str(self._db_path), stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

        print(f"[face_database] Saved {len(self._data)} enrolled faculty "
              f"to {self._db_path}")

    def add_face(self, name: str, embeddings: list) -> None:
        """Add or update a faculty member's embeddings.

        Args:
            name: Faculty member's name (used as the key).
            embeddings: A list of embedding vectors (numpy arrays), one per
                        capture. Storing multiple templates per person enables
                        majority-vote matching which dramatically reduces
                        false positives.

        Note: This does NOT automatically save to disk. Call save() after.
        """
        if name in self._data:
            print(f"[face_database] Updating existing entry for {name}")
        else:
            print(f"[face_database] Adding new entry for {name}")
        self._data[name] = [emb.astype(np.float32) for emb in embeddings]

    def remove_face(self, name: str) -> bool:
        """Remove a faculty member from the database.

        Args:
            name: Faculty member's name.

        Returns:
            True if the person was found and removed, False if not found.

        Note: This does NOT automatically save to disk. Call save() after.
        """
        if name in self._data:
            del self._data[name]
            print(f"[face_database] Removed {name}")
            return True
        else:
            print(f"[face_database] {name} not found in database")
            return False

    def get_embedding(self, name: str) -> Optional[List[np.ndarray]]:
        """Get a specific faculty member's embeddings.

        Args:
            name: Faculty member's name.

        Returns:
            A list of embedding arrays, or None if not enrolled.
        """
        return self._data.get(name)

    def get_all(self) -> Dict[str, List[np.ndarray]]:
        """Get all enrolled faculty and their embeddings.

        Returns:
            A dict mapping names to lists of embedding arrays.
        """
        return dict(self._data)

    def list_enrolled(self) -> List[str]:
        """Get the names of all enrolled faculty.

        Returns:
            A sorted list of enrolled names.
        """
        return sorted(self._data.keys())

    @property
    def count(self) -> int:
        """How many faculty are currently enrolled."""
        return len(self._data)

    def rotate_key(self, new_key_path: Optional[Path] = None) -> None:
        """Generate a new encryption key and re-encrypt the database.

        This is the incident response procedure described in
        PRIVACY_POLICY.md section 11: if the key is suspected to be
        compromised, rotate it immediately.

        Args:
            new_key_path: Where to save the new key. Defaults to the
                          same location, overwriting the old key.
        """
        target_path = new_key_path or self._key_path
        print(f"[face_database] Rotating encryption key...")

        # Generate new key.
        self._key = _generate_key(target_path)
        self._fernet = Fernet(self._key)

        # Re-encrypt and save with the new key.
        self.save()
        print(f"[face_database] Key rotation complete. "
              f"Database re-encrypted with new key.")