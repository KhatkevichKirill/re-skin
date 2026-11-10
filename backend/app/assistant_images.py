"""Bounded JPEG envelope validation without a native image decoder.

The trusted producer decodes each frame in the media sandbox before publishing
it. Transport validation checks its hash, dimensions and JPEG structure only;
it never loads Pillow/libjpeg inside a credential-bearing process.
"""
from __future__ import annotations


def read_jpeg_dimensions(raw: bytes) -> tuple[int, int]:
    if (not isinstance(raw, bytes) or not 4 <= len(raw) <= 512 * 1024
            or raw[:2] != b'\xff\xd8' or raw[-2:] != b'\xff\xd9'):
        raise ValueError('invalid bounded JPEG envelope')
    position, dimensions, scans, entropy_bytes = 2, None, 0, 0
    tables = set()
    while position < len(raw):
        if raw[position] != 255:
            raise ValueError('invalid JPEG marker')
        while position < len(raw) and raw[position] == 255:
            position += 1
        if position >= len(raw):
            raise ValueError('truncated JPEG marker')
        marker = raw[position]
        position += 1
        if marker == 0xD9:
            if position != len(raw) or dimensions is None or not scans or not entropy_bytes or tables != {0xDB, 0xC4}:
                raise ValueError('incomplete JPEG image')
            return dimensions
        if marker in (0, 0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            raise ValueError('unexpected JPEG marker')
        if position + 2 > len(raw):
            raise ValueError('truncated JPEG segment')
        size = int.from_bytes(raw[position:position + 2], 'big')
        if size < 2 or position + size > len(raw) - 2:
            raise ValueError('invalid JPEG segment size')
        payload = raw[position + 2:position + size]
        position += size
        if marker in (0xDB, 0xC4):
            if not payload:
                raise ValueError('empty JPEG table')
            tables.add(marker)
        if marker in (0xC0, 0xC2):
            if dimensions is not None or len(payload) < 6 or payload[0] != 8:
                raise ValueError('unsupported JPEG frame')
            height, width = int.from_bytes(payload[1:3], 'big'), int.from_bytes(payload[3:5], 'big')
            components = payload[5]
            if not 1 <= width <= 768 or not 1 <= height <= 768 or components not in (1, 3) or len(payload) != 6 + 3 * components:
                raise ValueError('JPEG dimensions or components exceed limits')
            dimensions = width, height
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            raise ValueError('unsupported JPEG coding')
        if marker == 0xDA:
            if dimensions is None or not payload or payload[0] not in (1, 3) or len(payload) != 4 + 2 * payload[0]:
                raise ValueError('invalid JPEG scan')
            scans += 1
            while position < len(raw) - 1:
                if raw[position] != 255:
                    entropy_bytes += 1
                    position += 1
                elif raw[position + 1] == 0 or 0xD0 <= raw[position + 1] <= 0xD7:
                    entropy_bytes += raw[position + 1] == 0
                    position += 2
                else:
                    break
    raise ValueError('missing JPEG end marker')
