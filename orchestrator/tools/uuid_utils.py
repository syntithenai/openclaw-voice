"""UUIDv7 generation utilities for sortable, collision-safe identifiers."""

import time
import secrets
import uuid


def generate_uuidv7() -> str:
    """
    Generate a UUIDv7 (timestamp-ordered UUID).
    
    Format: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx
    - First 48 bits: Unix timestamp in milliseconds
    - Version field: 0111 (7)
    - Variant field: 10
    - Remaining bits: random
    
    Returns:
        String representation of UUIDv7
    """
    # Get current timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)
    
    # Generate random bytes for the rest
    random_bytes = secrets.token_bytes(10)
    
    # Construct UUID bytes
    # First 48 bits: timestamp
    uuid_bytes = bytearray()
    uuid_bytes.extend(timestamp_ms.to_bytes(6, byteorder='big'))
    
    # Next 16 bits: version (4 bits = 0111) + random (12 bits)
    version_and_random = (0x7000 | (int.from_bytes(random_bytes[0:2], 'big') & 0x0FFF))
    uuid_bytes.extend(version_and_random.to_bytes(2, byteorder='big'))
    
    # Next 16 bits: variant (2 bits = 10) + random (14 bits)
    variant_and_random = (0x8000 | (int.from_bytes(random_bytes[2:4], 'big') & 0x3FFF))
    uuid_bytes.extend(variant_and_random.to_bytes(2, byteorder='big'))
    
    # Remaining 48 bits: random
    uuid_bytes.extend(random_bytes[4:10])
    
    # Convert to UUID and return string
    return str(uuid.UUID(bytes=bytes(uuid_bytes)))


def uuidv7_timestamp(uuid_str: str) -> float:
    """
    Extract timestamp from a UUIDv7 string.
    
    Args:
        uuid_str: UUIDv7 string
        
    Returns:
        Unix timestamp in seconds (float)
    """
    uuid_obj = uuid.UUID(uuid_str)
    uuid_bytes = uuid_obj.bytes
    
    # Extract first 48 bits (6 bytes) as timestamp in milliseconds
    timestamp_ms = int.from_bytes(uuid_bytes[0:6], byteorder='big')
    
    return timestamp_ms / 1000.0
