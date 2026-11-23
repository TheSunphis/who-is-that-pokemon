#!/usr/bin/env python3
"""Generate paired revealed/black-silhouette quiz images from a PokeAPI regional Pokédex.

The Alola run uses PokeAPI's ``updated-alola`` Pokédex and expands every listed species to all
PokeAPI varieties, so regional forms and Ultra Necrozma are included. Network and image work runs
concurrently; PokeAPI responses and sprites are cached under ``.cache/`` for repeatable reruns.

Dependencies: pip install pillow opencv-python numpy
Example: python3 scripts/generate_region.py --region alola --template assets/quiz-template.jpg
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "pokeapi"
USER_AGENT = "WhoIsThatPokemon/0.1 (PokeAPI client)"
POKEDEXES = {"alola": "updated-alola"}
REGIONS = ("kanto", "johto", "hoenn", "sinnoh", "unova", "kalos", "alola",
           "galar", "hisui", "paldea")
MAX_ARTWORK_BOX = (222, 188)
ARTWORK_CENTER = (138, 128)


def _cache_file(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.cache"


def get_bytes(url: str, *, refresh: bool = False) -> bytes:
    """Fetch a URL with a small on-disk cache and retries for transient API errors."""
    cache_path = _cache_file(url)
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,image/png,image/webp,*/*",
    }
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers=headers), timeout=35) as response:
                payload = response.read()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = cache_path.with_name(cache_path.name + f".{threading.get_ident()}.tmp")
            temp.write_bytes(payload)
            temp.replace(cache_path)
            return payload
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 ** attempt
            time.sleep(min(delay, 20))
        except (TimeoutError, URLError, OSError):
            if attempt == 4:
                raise
            time.sleep(min(1.5 ** attempt, 20))
    raise RuntimeError(f"Could not fetch {url}")


def get_json(url: str, *, refresh: bool = False) -> dict:
    return json.loads(get_bytes(url, refresh=refresh))


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.casefold()).strip("-")
    return cleaned or "pokemon"


def find_old_silhouette(image: Image.Image) -> np.ndarray:
    """Find the large dark mystery figure on the left, avoiding the question mark and logo."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    x_end, y_end = min(width, int(width * 0.56)), min(height, int(height * 0.89))
    dark = (np.max(rgb[:y_end, :x_end], axis=2) < 35).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if count <= 1:
        raise ValueError("No black silhouette found in the left side of the template.")
    cx, cy = min(x_end - 1, int(width * 0.29)), min(y_end - 1, int(height * 0.48))
    label = int(labels[cy, cx])
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.uint8) * 255
    if cv2.countNonZero(component) < 1000:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = (labels == label).astype(np.uint8) * 255
    component = cv2.dilate(
        component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:y_end, :x_end] = component
    return mask


def clean_background(template: Image.Image) -> Image.Image:
    rgb = np.asarray(template.convert("RGB"))
    mask = find_old_silhouette(template)
    cleaned = cv2.inpaint(rgb, mask, 7, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned, mode="RGB")


def pokemon_art_url(record: dict) -> str | None:
    sprites = record.get("sprites") or {}
    other = sprites.get("other") or {}
    for url in (
        (other.get("official-artwork") or {}).get("front_default"),
        (other.get("home") or {}).get("front_default"),
        sprites.get("front_default"),
    ):
        if url:
            return str(url)
    return None


def crop_and_fit(artwork: Image.Image, canvas_size: tuple[int, int], *,
                 silhouette: bool = False) -> tuple[Image.Image, tuple[int, int]]:
    art = artwork.convert("RGBA")
    alpha = art.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value >= 24 else 0).getbbox()
    if bbox is None:
        raise ValueError("Sprite has no visible artwork.")
    art = art.crop(bbox)
    if silhouette:
        alpha = art.getchannel("A").point(lambda value: 255 if value >= 64 else 0)
        black = Image.new("RGBA", art.size, (0, 0, 0, 255))
        black.putalpha(alpha)
        art = black
    scale = min(MAX_ARTWORK_BOX[0] / art.width, MAX_ARTWORK_BOX[1] / art.height)
    size = (max(1, round(art.width * scale)), max(1, round(art.height * scale)))
    art = art.resize(size, Image.Resampling.LANCZOS)
    if silhouette:
        art.putalpha(art.getchannel("A").filter(ImageFilter.GaussianBlur(0.25)))
    position = (round(ARTWORK_CENTER[0] - size[0] / 2),
                round(ARTWORK_CENTER[1] - size[1] / 2))
    return art, position


def compose(background: Image.Image, artwork: Image.Image, *, silhouette: bool) -> Image.Image:
    sprite, position = crop_and_fit(artwork, background.size, silhouette=silhouette)
    result = background.convert("RGBA").copy()
    if not silhouette:
        shadow_alpha = sprite.getchannel("A").filter(ImageFilter.GaussianBlur(2.0))
        shadow_alpha = shadow_alpha.point(lambda value: round(value * 0.42))
        shadow = Image.new("RGBA", sprite.size, (76, 45, 137, 0))
        shadow.putalpha(shadow_alpha)
        result.alpha_composite(shadow, dest=(position[0] + 2, position[1] + 2))
    result.alpha_composite(sprite, dest=position)
    return result.convert("RGB")


def _save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=93, optimize=True, progressive=True, subsampling=0)


def _pokemon_details(form: dict, *, refresh: bool) -> dict:
    name = form["name"]
    url = f"https://pokeapi.co/api/v2/pokemon/{name}/"
    return get_json(url, refresh=refresh)


def _generate_form(form: dict, background: Image.Image, region_dir: Path,
                   *, refresh: bool) -> dict:
    name = form["name"]
    details = _pokemon_details(form, refresh=refresh)
    sprite_url = pokemon_art_url(details)
    if not sprite_url:
        return {**form, "status": "missing_sprite", "sprite_url": None}
    raw = get_bytes(sprite_url, refresh=refresh)
    artwork = Image.open(BytesIO(raw)).convert("RGBA")
    stem = slug(name)
    silhouette_path = region_dir / "silhouettes" / f"{stem}.jpg"
    revealed_path = region_dir / "revealed" / f"{stem}.jpg"
    _save_jpeg(compose(background, artwork, silhouette=True), silhouette_path)
    _save_jpeg(compose(background, artwork, silhouette=False), revealed_path)
    return {
        **form,
        "status": "generated",
        "sprite_url": sprite_url,
        "silhouette": silhouette_path.relative_to(ROOT).as_posix(),
        "revealed": revealed_path.relative_to(ROOT).as_posix(),
    }


def list_forms(region: str, *, workers: int, refresh: bool) -> tuple[dict, list[dict]]:
    pokedex_name = POKEDEXES.get(region)
    if pokedex_name is None:
        raise ValueError(f"Region {region!r} is not configured yet; currently supported: alola.")
    pokedex = get_json(f"https://pokeapi.co/api/v2/pokedex/{pokedex_name}/", refresh=refresh)
    species_entries = pokedex["pokemon_entries"]
    ordered_entries = sorted(species_entries, key=lambda row: row["entry_number"])

    def fetch_species(entry: dict) -> tuple[int, str, dict]:
        species = entry["pokemon_species"]
        record = get_json(species["url"], refresh=refresh)
        return entry["entry_number"], species["name"], record

    species_data = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_species, entry) for entry in ordered_entries]
        for future in as_completed(futures):
            species_data.append(future.result())
    species_data.sort(key=lambda item: item[0])

    forms = []
    seen = set()
    for dex_number, species_name, record in species_data:
        for variety in record.get("varieties", []):
            pokemon = variety["pokemon"]
            pokemon_name = pokemon["name"]
            if pokemon_name in seen:
                continue
            seen.add(pokemon_name)
            forms.append({
                "dex_number": dex_number,
                "species": species_name,
                "name": pokemon_name,
                "is_default": bool(variety.get("is_default")),
            })
    return pokedex, forms


def generate(region: str, template_path: Path, *, workers: int = 8,
             limit: int | None = None, refresh: bool = False) -> dict:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    region_dir = ROOT / "regions" / region
    image = Image.open(template_path).convert("RGB")
    background = clean_background(image)
    pokedex, forms = list_forms(region, workers=workers, refresh=refresh)
    if limit is not None:
        forms = forms[:max(0, limit)]
    print(f"PokeAPI {pokedex['name']}: {len(pokedex['pokemon_entries'])} Dex entries, "
          f"{len(forms)} distinct forms to render with {workers} workers.", flush=True)

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate_form, form, background, region_dir,
                                  refresh=refresh): form for form in forms}
        for future in as_completed(futures):
            form = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # preserve partial progress and explain any PokeAPI failures
                result = {**form, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            completed += 1
            if completed % 25 == 0 or completed == len(forms):
                print(f"Rendered {completed}/{len(forms)} forms", flush=True)

    results.sort(key=lambda row: (row["dex_number"], row["name"]))
    manifest = {
        "region": region,
        "pokedex": pokedex["name"],
        "pokedex_url": f"https://pokeapi.co/api/v2/pokedex/{pokedex['name']}/",
        "species_count": len(pokedex["pokemon_entries"]),
        "form_count": len(forms),
        "generated_count": sum(row["status"] == "generated" for row in results),
        "missing_or_failed_count": sum(row["status"] != "generated" for row in results),
        "form_policy": "all PokeAPI varieties for every species in the updated regional Pokédex",
        "results": results,
    }
    region_dir.mkdir(parents=True, exist_ok=True)
    (region_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (region_dir / "README.md").write_text(
        f"# {region.title()} Pokémon quiz images\n\n"
        f"Source Pokédex: [`{pokedex['name']}`]({manifest['pokedex_url']}) "
        f"({manifest['species_count']} entries).\n\n"
        f"The generator expands every listed species to all PokeAPI varieties. It rendered "
        f"{manifest['generated_count']} forms as paired images in `silhouettes/` and `revealed/`. "
        f"This includes Alolan forms and battle forms such as Ultra Necrozma when PokeAPI provides "
        f"official artwork. See `manifest.json` for each file and source URL.\n\n"
        "Pokémon names and artwork are the property of their respective rights holders; artwork "
        "is fetched from PokeAPI's sprite data.\n",
        encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="alola", choices=sorted(POKEDEXES),
                        help="regional Pokédex to process (currently: alola)")
    parser.add_argument("--template", type=Path,
                        default=ROOT / "assets" / "quiz-template.jpg",
                        help="mystery-quiz image to reuse as the background")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel API/image workers (default: 8)")
    parser.add_argument("--limit", type=int,
                        help="optional maximum number of forms, useful for a quick test")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore HTTP cache and refetch PokeAPI metadata and sprites")
    args = parser.parse_args()
    manifest = generate(args.region, args.template, workers=args.workers,
                        limit=args.limit, refresh=args.refresh)
    print(f"Done: {manifest['generated_count']} generated; "
          f"{manifest['missing_or_failed_count']} missing/failed. "
          f"See regions/{args.region}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
