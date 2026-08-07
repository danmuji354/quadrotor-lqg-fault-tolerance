import json
from pathlib import Path

from PIL import Image

from quadrotor_lqg.gallery import gallery_contract


def test_gallery_contract_has_all_asset_roles():
    result = {"position_rmse_m": 0.01, "final_position_error_m": 0.02}
    contract = gallery_contract({"nominal": result, "GPS dropout": result})
    assert contract["schema_version"] == 1
    assert {asset["role"] for asset in contract["assets"]} == {
        "hero",
        "analysis",
        "animation",
        "diagram",
    }


def test_checked_in_gallery_matches_website_contract():
    gallery = Path(__file__).parents[1] / "artifacts" / "gallery"
    contract = json.loads((gallery / "showcase.json").read_text())
    for asset in contract["assets"]:
        path = gallery / asset["path"]
        assert path.is_file()
        if path.suffix.lower() in {".png", ".gif"}:
            with Image.open(path) as image:
                assert image.size == (asset["width"], asset["height"])
