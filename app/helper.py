import base64
import zlib


def decompress_image_to_base64(compressed_binary):
    """
    Return base64 string from a zlib-compressed Binary field or None on
    failure. Also supports arrays of Binary objects (returns list of str).
    """
    if not compressed_binary:
        return None
    try:
        # If a list/array is provided, decompress each element
        if isinstance(compressed_binary, list):
            out = []
            for item in compressed_binary:
                try:
                    raw = zlib.decompress(bytes(item))
                    out.append(base64.b64encode(raw).decode('utf-8'))
                except Exception:
                    out.append(None)
            return out
        # Single Binary
        raw_bytes = zlib.decompress(bytes(compressed_binary))
        return base64.b64encode(raw_bytes).decode('utf-8')
    except Exception:
        return None

