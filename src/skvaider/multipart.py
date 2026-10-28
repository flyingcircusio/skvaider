"""Helpers for multipart/form-data request bodies.

The proxy has to route requests that carry an audio file (for example
transcription requests) without parsing or re-encoding the body: parsing the
upload is the backend's job. Only the plain form fields that the proxy itself
needs, like ``model`` and ``stream``, are extracted here.
"""

from email.parser import BytesParser
from email.policy import HTTP


def text_fields(body: bytes, content_type: str) -> dict[str, str]:
    """Return the plain (non-file) fields of a multipart/form-data body.

    Returns an empty mapping when the body is not multipart or cannot be
    parsed. File fields are ignored: their payload is binary and must not be
    decoded, and callers must forward the original body to the backend.
    """
    if "multipart/form-data" not in content_type.lower():
        return {}
    try:
        message = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: "
            + content_type.encode("latin-1", errors="replace")
            + b"\r\n\r\n"
            + body
        )
    except Exception:
        return {}
    if not message.is_multipart():
        return {}

    fields: dict[str, str] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name or part.get_filename():
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        fields[str(name)] = payload.decode("utf-8", errors="replace")
    return fields
