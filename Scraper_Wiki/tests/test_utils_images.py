import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scraper_wiki as sw


def test_extract_images_basic():
    html = (
        "<div class='thumb'>"
        "<img src='//upload.wikimedia.org/img.jpg'/>"
        "<div class='thumbcaption'>Cap</div>"
        "</div>"
    )
    res = sw.extract_images(html)
    assert res == [{"image_url": "https://upload.wikimedia.org/img.jpg", "caption": "Cap"}]


def test_extract_images_no_caption():
    html = "<figure><img src='pic.png'/></figure>"
    res = sw.extract_images(html)
    assert res == [{"image_url": "pic.png", "caption": ""}]


def test_extract_images_multiple():
    html = (
        "<div class='thumb'>"
        "<img src='img1.png'/>"
        "<div class='thumbcaption'>One</div>"
        "</div>"
        "<figure><img src='//example.com/img2.jpg'/><figcaption>Two</figcaption></figure>"
    )
    res = sw.extract_images(html)
    assert res == [
        {"image_url": "img1.png", "caption": "One"},
        {"image_url": "https://example.com/img2.jpg", "caption": "Two"},
    ]
