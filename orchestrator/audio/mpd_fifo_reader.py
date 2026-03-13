import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass


logger = logging.getLogger("orchestrator.audio.mpd_fifo_reader")


@dataclass
class MPDFifoStats:
    fifo_path: str
    sample_rate: int
    channels: int
    bits_per_sample: int
    bytes_read: int
    chunks_read: int
    writer_disconnects: int


class MPDFifoReader:
    """Scaffold for reading raw PCM from MPD FIFO output.

    This intentionally does not mix into orchestrator playback yet.
    It provides lifecycle + ingest plumbing for the next implementation phase.
    """

    def __init__(
        self,
        fifo_path: str,
        sample_rate: int,
        channels: int,
        bits_per_sample: int = 16,
        chunk_bytes: int = 4096,
        max_cached_chunks: int = 128,
    ) -> None:
        self.fifo_path = fifo_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.chunk_bytes = max(256, chunk_bytes)
        self._fd: int | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._bytes_read = 0
        self._chunks_read = 0
        self._writer_disconnects = 0
        self._chunk_cache: deque[bytes] = deque(maxlen=max(1, max_cached_chunks))

    async def start(self) -> None:
        if self._running:
            return

        fifo_dir = os.path.dirname(self.fifo_path)
        if fifo_dir:
            os.makedirs(fifo_dir, exist_ok=True)
            try:
                os.chmod(fifo_dir, 0o777)
            except Exception:
                pass

        if not os.path.exists(self.fifo_path):
            os.mkfifo(self.fifo_path, 0o666)
            try:
                os.chmod(self.fifo_path, 0o666)
            except Exception:
                pass
            logger.info("Created MPD FIFO path: %s", self.fifo_path)

        self._fd = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        self._running = True
        self._task = asyncio.create_task(self._read_loop())
        logger.info(
            "MPD FIFO reader scaffold started (path=%s, format=%d:%d:%d)",
            self.fifo_path,
            self.sample_rate,
            self.bits_per_sample,
            self.channels,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    async def _read_loop(self) -> None:
        assert self._fd is not None
        while self._running:
            try:
                chunk = await asyncio.to_thread(os.read, self._fd, self.chunk_bytes)
                if not chunk:
                    self._writer_disconnects += 1
                    await asyncio.sleep(0.05)
                    continue

                self._chunk_cache.append(chunk)
                self._bytes_read += len(chunk)
                self._chunks_read += 1

                if self._chunks_read % 200 == 0:
                    logger.info(
                        "MPD FIFO ingest scaffold: chunks=%d bytes=%d",
                        self._chunks_read,
                        self._bytes_read,
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("MPD FIFO read error: %s", exc)
                await asyncio.sleep(0.2)

    def get_stats(self) -> MPDFifoStats:
        return MPDFifoStats(
            fifo_path=self.fifo_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
            bits_per_sample=self.bits_per_sample,
            bytes_read=self._bytes_read,
            chunks_read=self._chunks_read,
            writer_disconnects=self._writer_disconnects,
        )

    def pop_chunk(self) -> bytes | None:
        if not self._chunk_cache:
            return None
        return self._chunk_cache.popleft()
