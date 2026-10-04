"""Validate local page references and the published figure manifest."""
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for name in ("src", "href"):
            if name in attrs:
                self.references.append(attrs[name])


parser = References()
parser.feed((SITE / "index.html").read_text())
for reference in parser.references:
    if reference.startswith("#"):
        assert not reference[1:] or reference[1:] in parser.ids, reference
    elif not reference.startswith(("https:", "http:", "mailto:", "data:")):
        assert (SITE / reference.split("#")[0]).is_file(), reference
manifest = json.loads((SITE / "assets/provenance.json").read_text())
assert len(manifest) == 28, "Expected four FFHQ grids and 24 trajectory figures"
for entry in manifest:
    asset = SITE / "assets" / entry["asset"]
    assert asset.is_file(), asset
    expected = entry.get("asset_sha256", entry["source_sha256"])
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == expected, asset
for dataset in ("moon", "checkerboard_grid", "letter_f", "letter_m"):
    for method in ("drift_flow_matching_steps_1", "drift_flow_matching_steps_20", "flow_matching", "mean_flow_steps_1", "mean_flow_steps_20", "drift"):
        assert (SITE / "assets" / f"{dataset}-{method}.jpg").is_file()
print("Validated page references and all 28 archived figures.")
