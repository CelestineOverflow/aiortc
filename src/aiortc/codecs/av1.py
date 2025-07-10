import fractions
import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import tee
from struct import pack, unpack_from
from typing import Optional, Type, TypeVar, cast

import av
from av.error import FFmpegError
from av.frame import Frame
from av.packet import Packet
from av.video.codeccontext import VideoCodecContext

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Decoder, Encoder

logger = logging.getLogger(__name__)

DEFAULT_BITRATE = 1_000_000
MIN_BITRATE = 500_000
MAX_BITRATE = 10_000_000

MAX_FRAME_RATE = 30
PACKET_MAX = 1300

NAL_TYPE_FU_A = 28
NAL_TYPE_STAP_A = 24

NAL_HEADER_SIZE = 1
FU_A_HEADER_SIZE = 2
LENGTH_FIELD_SIZE = 2
STAP_A_HEADER_SIZE = NAL_HEADER_SIZE + LENGTH_FIELD_SIZE

DESCRIPTOR_T = TypeVar("DESCRIPTOR_T", bound="AV1PayloadDescriptor")
T = TypeVar("T")


def pairwise(iterable: Sequence[T]) -> Iterator[tuple[T, T]]:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class AV1PayloadDescriptor:
    """Parse RTP payloads that carry Annex-B style AV1 NALUs."""

    def __init__(self, first_fragment: bool) -> None:
        self.first_fragment = first_fragment

    def __repr__(self) -> str:  # pragma: no cover
        return f"AV1PayloadDescriptor(FF={self.first_fragment})"

    @classmethod
    def parse(cls: Type[DESCRIPTOR_T], data: bytes) -> tuple[DESCRIPTOR_T, bytes]:
        output = b""
        if len(data) < 2:
            raise ValueError("NAL unit is too short")

        nal_type = data[0] & 0x1F
        f_nri = data[0] & 0xE0
        pos = NAL_HEADER_SIZE

        if 1 <= nal_type <= 23:
            # Single NALU
            output = b"\x00\x00\x00\x01" + data
            obj = cls(True)
        elif nal_type == NAL_TYPE_FU_A:
            # Fragmentation unit
            first_fragment = bool(data[pos] & 0x80)
            original_nal_type = data[pos] & 0x1F
            pos += 1
            if first_fragment:
                output += b"\x00\x00\x00\x01" + bytes([f_nri | original_nal_type])
            output += data[pos:]
            obj = cls(first_fragment)
        elif nal_type == NAL_TYPE_STAP_A:
            # Aggregation packet
            offsets: list[int] = []
            while pos < len(data):
                if len(data) < pos + LENGTH_FIELD_SIZE:
                    raise ValueError("STAP-A length field truncated")
                nalu_size = unpack_from("!H", data, pos)[0]
                pos += LENGTH_FIELD_SIZE
                offsets.append(pos)
                pos += nalu_size
                if len(data) < pos:
                    raise ValueError("STAP-A data truncated")
            offsets.append(len(data) + LENGTH_FIELD_SIZE)
            for start, end in pairwise(offsets):
                end -= LENGTH_FIELD_SIZE
                output += b"\x00\x00\x00\x01" + data[start:end]
            obj = cls(True)
        else:
            raise ValueError(f"NAL type {nal_type} not supported")

        return obj, output


class AV1Decoder(Decoder):
    """Tiny wrapper that prefers libdav1d when available."""

    def __init__(self) -> None:
        try:
            self.codec = av.CodecContext.create("libdav1d", "r")
            logger.info("Using libdav1d decoder")
        except FFmpegError:
            logger.warning("libdav1d unavailable, falling back to built-in decoder")
            self.codec = av.CodecContext.create("av1", "r")

    def decode(self, encoded_frame: JitterFrame) -> list[Frame]:
        try:
            pkt = av.Packet(encoded_frame.data)
            pkt.pts = encoded_frame.timestamp
            pkt.time_base = VIDEO_TIME_BASE
            return cast(list[Frame], self.codec.decode(pkt))
        except FFmpegError as e:
            logger.warning("Decode failed, skipping: %s", e)
            return []


class AV1Encoder(Encoder):
    """Packetises AV1 bit-stream into RTP-friendly NAL units."""

    def __init__(self) -> None:
        self.buffer_data = b""
        self.buffer_pts: Optional[int] = None
        self.codec: Optional[VideoCodecContext] = None
        self.__target_bitrate = DEFAULT_BITRATE

    # ---------------------------------------------------------------------
    # Packetisation helpers (FU-A, STAP-A, Annex-B splitter)
    # ---------------------------------------------------------------------
    @staticmethod
    def _packetize_fu_a(data: bytes) -> list[bytes]:
        available = PACKET_MAX - FU_A_HEADER_SIZE
        payload = data[NAL_HEADER_SIZE:]
        size = len(payload)
        num = math.ceil(size / available)
        large = size % num
        chunk = size // num

        f_nri = data[0] & 0xE0
        nal = data[0] & 0x1F
        fu_indicator = f_nri | NAL_TYPE_FU_A

        start_hdr = bytes([fu_indicator, nal | 0x80])
        mid_hdr = bytes([fu_indicator, nal])
        end_hdr = bytes([fu_indicator, nal | 0x40])

        pkts: list[bytes] = []
        off = 0
        hdr = start_hdr
        for i in range(num):
            part = payload[off : off + chunk + (1 if i < large else 0)]
            off += len(part)
            if i == num - 1:
                hdr = end_hdr
            pkts.append(hdr + part)
            hdr = mid_hdr
        return pkts

    @staticmethod
    def _packetize_stap_a(data: bytes, it: Iterator[bytes]) -> tuple[bytes, Optional[bytes]]:
        counter = 0
        available = PACKET_MAX - STAP_A_HEADER_SIZE
        stap_header = (data[0] & 0xE0) | NAL_TYPE_STAP_A
        payload = b""

        try:
            nalu = data
            while len(nalu) <= available and counter < 9:
                # keep track of key-frame bit & NRI merges
                stap_header |= nalu[0] & 0x80
                nri = nalu[0] & 0x60
                if stap_header & 0x60 < nri:
                    stap_header = (stap_header & 0x9F) | nri

                available -= LENGTH_FIELD_SIZE + len(nalu)
                counter += 1
                payload += pack("!H", len(nalu)) + nalu
                nalu = next(it)

            if counter == 0:
                # Could not aggregate, send single
                return data, nalu
        except StopIteration:
            nalu = None

        if counter <= 1:
            return data, nalu
        return bytes([stap_header]) + payload, nalu

    @staticmethod
    def _split_bitstream(buf: bytes) -> Iterator[bytes]:
        i = 0
        while True:
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                return
            i += 3
            start = i
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                yield buf[start:]
                return
            # Detect 4-byte start codes (…00 00 00 01)
            yield buf[start : (i - 1 if buf[i - 1] == 0 else i)]

    # ------------------------------------------------------------------
    # Public encode helpers
    # ------------------------------------------------------------------
    @classmethod
    def _packetize(cls, nalus: Iterable[bytes]) -> list[bytes]:
        pkts: list[bytes] = []
        it = iter(nalus)
        nalu = next(it, None)
        while nalu is not None:
            if len(nalu) > PACKET_MAX:
                pkts.extend(cls._packetize_fu_a(nalu))
                nalu = next(it, None)
            else:
                stap, nalu = cls._packetize_stap_a(nalu, it)
                pkts.append(stap)
        return pkts

    # ------------------------------------------------------------------
    # Codec handling
    # ------------------------------------------------------------------
    def _ensure_codec(self, frame: av.VideoFrame) -> None:
        need_new = (
            self.codec is None
            or frame.width != self.codec.width
            or frame.height != self.codec.height
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
        )
        if not need_new:
            return

        self.buffer_data = b""
        self.buffer_pts = None
        self.codec = av.CodecContext.create("libaom-av1", "w")
        self.codec.width = frame.width
        self.codec.height = frame.height
        self.codec.bit_rate = self.target_bitrate
        self.codec.pix_fmt = "yuv420p"
        self.codec.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
        self.codec.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
        self.codec.options = {"level": "31", "tune": "zerolatency"}

    # ------------------------------------------------------------------
    # Encoding pipeline
    # ------------------------------------------------------------------
    def _encode_frame(self, frame: av.VideoFrame, force_keyframe: bool) -> Iterator[bytes]:
        self._ensure_codec(frame)
        frame.pict_type = (
            av.video.frame.PictureType.I if force_keyframe else av.video.frame.PictureType.NONE
        )
        for pkt in self.codec.encode(frame):
            yield from self._split_bitstream(bytes(pkt))

    def encode(self, frame: Frame, force_keyframe: bool = False) -> tuple[list[bytes], int]:
        assert isinstance(frame, av.VideoFrame)
        nalus = self._encode_frame(frame, force_keyframe)
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        return self._packetize(nalus), timestamp

    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        assert isinstance(packet, av.Packet)
        nalus = self._split_bitstream(bytes(packet))
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        return self._packetize(nalus), timestamp

    # ------------------------------------------------------------------
    # Bit-rate interface
    # ------------------------------------------------------------------
    @property
    def target_bitrate(self) -> int:
        """Target bitrate in bit/s (clamped)."""
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        self.__target_bitrate = max(MIN_BITRATE, min(bitrate, MAX_BITRATE))


# ==========================================================================
# Utility for depayloading RTP — handy in tests or packet reassembly helpers
# ==========================================================================

def AV1_depayload(payload: bytes) -> bytes:
    """Strip RTP payload header and return Annex-B NALU stream."""
    _, data = AV1PayloadDescriptor.parse(payload)
    return data
