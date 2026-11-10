"""Transport checks are pure Python; native pixel validation belongs in jail."""
import io

import pytest
from PIL import Image

from app.assistant_images import read_jpeg_dimensions


def jpeg(*, size=(24, 16), progressive=False):
    buffer = io.BytesIO()
    Image.new('RGB', size, 'red').save(buffer, format='JPEG', progressive=progressive)
    return buffer.getvalue()


@pytest.mark.parametrize('progressive', [False, True])
def test_normal_jpeg_envelope_does_not_invoke_native_decoder(monkeypatch, progressive):
    raw = jpeg(progressive=progressive)
    monkeypatch.setattr(Image, 'open', lambda *args, **kwargs: pytest.fail('native decoder invoked'))
    assert read_jpeg_dimensions(raw) == (24, 16)


@pytest.mark.parametrize('bad', [
    b'\xff\xd8\xff\xd9', b'\xff\xd8\xff\xe0\xff\xff\xff\xd9',
    b'\xff\xd8\xff\xda\x00\x02\xff\xd9',
])
def test_truncated_or_missing_structure_is_rejected(bad):
    with pytest.raises(ValueError):
        read_jpeg_dimensions(bad)


def test_geometry_byte_budget_and_trailing_payload_are_rejected():
    for raw in (jpeg(size=(769, 2)), jpeg() + b'private', b'\xff\xd8' + b'X' * (512 * 1024) + b'\xff\xd9'):
        with pytest.raises(ValueError):
            read_jpeg_dimensions(raw)
