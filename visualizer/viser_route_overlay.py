"""Isolated route overlay for the installed Viser 1.1.1 client; no geometry edits."""
import re
from pathlib import Path
from unittest.mock import patch

import viser
import zstandard  # Already required by Viser.


def overlay_client(destination):
    if viser.__version__ != "1.1.1":
        raise RuntimeError("Review the route overlay against this Viser version first")
    page = (Path(viser.__file__).parent / "client/build/index.html").read_text()
    encoded = re.search(r'data-p="([^"]+)"', page).group(1)
    alphabet = "!#$%()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_abcdefghijklmnopqrstuvwxyz{|}~"
    packed = bytearray()
    for start in range(0, len(encoded), 5):
        chunk = encoded[start:start + 5]
        value = 0
        for char in chunk:
            value = value * 88 + alphabet.index(char)
        packed.extend(value.to_bytes(4 if len(chunk) == 5 else len(chunk) - 1, "big"))
    raw = zstandard.ZstdDecompressor().decompress(packed)
    css_size = int(re.search(r'data-ss="(\d+)"', page).group(1))
    script = raw[css_size:].decode()
    replacements = {
        "UR=b.forwardRef(function({props:e,children:t},n)":
            "UR=b.forwardRef(function({props:e,children:t,name:routeName},n)",
        "points:r,lineWidth:e.thickness,worldUnits:e.thickness_units===`world`,color:a,vertexColors:o,segments:!0":
            "points:r,lineWidth:e.thickness,worldUnits:e.thickness_units===`world`,color:a,vertexColors:o,segments:!0,"
            "ref:object=>{if(object&&routeName?.startsWith('/dataset/route/')){"
            "object.renderOrder=routeName.endsWith('/outline')?10000:10001;"
            "object.material.depthTest=false;object.material.depthWrite=false;"
            "object.material.transparent=true;object.material.fog=false;"
            "object.material.needsUpdate=true;}}",
    }
    for original, replacement in replacements.items():
        if script.count(original) != 1:
            raise RuntimeError("Viser client changed; route overlay needs review")
        script = script.replace(original, replacement)
    # Keep Viser's original decompressor and its shared WASM module intact.
    compressed = zstandard.ZstdCompressor().compress(raw[:css_size] + script.encode())
    chunks = []
    for start in range(0, len(compressed), 4):
        chunk = compressed[start:start + 4]
        value = int.from_bytes(chunk, "big")
        digits = []
        for _ in range(len(chunk) + 1):
            digits.append(alphabet[value % 88])
            value //= 88
        chunks.append("".join(reversed(digits)))
    page = page.replace('data-p="' + encoded + '"', 'data-p="' + "".join(chunks) + '"', 1)
    page = re.sub(r'data-cs="\d+"', f'data-cs="{len(script.encode())}"', page, count=1)
    destination = Path(destination)
    destination.mkdir(exist_ok=True)
    (destination / "index.html").write_text(page)
    return destination


def create_server(*, host, port):
    client = overlay_client(Path(__file__).parent / "viewer-client")
    class ViewerSocket(viser.infra.WebsockServer):
        def __init__(self, *args, **kwargs):
            kwargs["http_server_root"] = client
            super().__init__(*args, **kwargs)
    with patch.object(viser.infra, "WebsockServer", ViewerSocket):
        return viser.ViserServer(host=host, port=port)
