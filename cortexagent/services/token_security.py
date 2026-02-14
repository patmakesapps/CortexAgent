from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re


_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[a-z0-9\-._~+/]+=*")
_TOKENISH_PATTERN = re.compile(r"(?i)\b(access|refresh|id)_token\b[^,\n]*")


def redact_sensitive_text(value: str | None) -> str:
    if not value:
        return ""
    out = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    out = _TOKENISH_PATTERN.sub("[REDACTED_TOKEN_FIELD]", out)
    return out


class TokenCipher:
    PREFIX = "enc:v1:"

    def __init__(self, secret: str | None) -> None:
        self._secret = (secret or "").strip()
        self._key = hashlib.sha256(self._secret.encode("utf-8")).digest() if self._secret else b""

    def enabled(self) -> bool:
        return bool(self._key)

    def encrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        if not self.enabled():
            return value
        raw = value.encode("utf-8")
        key = self._derive_keystream(len(raw))
        masked = bytes([left ^ right for left, right in zip(raw, key)])
        return f"{self.PREFIX}{base64.urlsafe_b64encode(masked).decode('ascii')}"

    def decrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        if not self.enabled():
            return value
        if not value.startswith(self.PREFIX):
            return value
        encoded = value[len(self.PREFIX) :]
        try:
            masked = base64.urlsafe_b64decode(encoded.encode("ascii"))
            key = self._derive_keystream(len(masked))
            raw = bytes([left ^ right for left, right in zip(masked, key)])
            return raw.decode("utf-8")
        except Exception:
            return None

    def _derive_keystream(self, length: int) -> bytes:
        # HMAC-SHA256 stream by counter blocks, sufficient for at-rest obfuscation.
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hmac.new(
                self._key,
                msg=str(counter).encode("ascii"),
                digestmod=hashlib.sha256,
            ).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])


def build_token_cipher_from_env() -> TokenCipher:
    secret = os.getenv("CONNECTED_ACCOUNTS_TOKEN_ENCRYPTION_KEY")
    return TokenCipher(secret=secret)
