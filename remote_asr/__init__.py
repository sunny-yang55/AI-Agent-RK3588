"""Remote ASR bridge used by the first-generation RK3588 prototype."""

from .client import RemoteASRClient
from .server import create_server, decode_wav

__all__ = ["RemoteASRClient", "create_server", "decode_wav"]
