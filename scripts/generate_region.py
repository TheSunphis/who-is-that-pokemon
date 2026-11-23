#!/usr/bin/env python3
"""Generate paired revealed/black-silhouette quiz images from PokeAPI regional Pokédexes.

Each regional collection expands its Pokédex species to all PokeAPI varieties. Split and DLC
Pokédexes are combined where relevant, so forms such as Ultra Necrozma are included. Network and
image work runs concurrently; PokeAPI responses and sprites are cached under ``.cache/``.

Dependencies: pip install pillow opencv-python numpy
Examples:
  python3 scripts/generate_region.py --region alola --template assets/quiz-template.jpg
  python3 scripts/generate_region.py --region all --workers 20
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "pokeapi"
USER_AGENT = "WhoIsThatPokemon/0.1 (PokeAPI client)"
REGION_POKEDEXES = {
    "kanto": ("kanto", "letsgo-kanto"),
    "johto": ("updated-johto",),
    "hoenn": ("updated-hoenn",),
    "sinnoh": ("extended-sinnoh",),
    "unova": ("updated-unova",),
    "kalos": ("kalos-central", "kalos-coastal", "kalos-mountain",
              "lumiose-city", "hyperspace"),
    "alola": ("updated-alola",),
    "galar": ("galar", "isle-of-armor", "crown-tundra"),
    "hisui": ("hisui",),
    "paldea": ("paldea", "kitakami", "blueberry"),
}
REGIONS = tuple(REGION_POKEDEXES)
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


def _english_name(species_record: dict, fallback: str) -> str:
    return next((entry["name"] for entry in species_record.get("names", [])
                 if entry.get("language", {}).get("name") == "en"),
                fallback.replace("-", " ").title())


def pokemon_display_name(species_name: str, pokemon_name: str, species_label: str) -> str:
    """Turn PokeAPI form slugs into readable card names such as ``Ultra Necrozma``."""
    if pokemon_name == species_name:
        return species_label
    suffix = pokemon_name[len(species_name):].strip("-") if pokemon_name.startswith(
        species_name + "-") else pokemon_name
    parts = suffix.split("-") if suffix else []
    regions = {"alola": "Alolan", "galar": "Galarian", "hisui": "Hisuian",
               "paldea": "Paldean"}
    region = next((part for part in parts if part in regions), None)
    if "ultra" in parts:
        return f"Ultra {species_label}"
    if "mega" in parts:
        extra = " X" if "x" in parts else " Y" if "y" in parts else ""
        return f"Mega {species_label}{extra}"
    if "gmax" in parts or "gigantamax" in parts:
        return f"Gigantamax {species_label}"
    if region:
        regional = f"{regions[region]} {species_label}"
        if "totem" in parts:
            return f"Totem {regional}"
        extras = [part for part in parts if part not in (region, "totem")]
        return regional if not extras else f"{regional} {' '.join(extras).title()}"
    if "totem" in parts:
        return f"Totem {species_label}"
    return f"{species_label} {' '.join(parts).title()}"


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


def remove_question_mark(image: Image.Image) -> Image.Image:
    """Reconstruct the template's question-mark area for revealed cards.

    The complete symbol includes yellow faces, blue outlines, and a dark drop shadow, so a
    color-only mask leaves noticeable fragments. This template has the mark at a fixed location;
    inpaint a slightly wider rectangular patch to remove all layers cleanly. The coordinates and
    inpaint radius scale with the supplied 480-by-270 template.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    x0, x1 = round(width * 307 / 480), round(width * 391 / 480)
    y0, y1 = round(height * 43 / 270), round(height * 170 / 270)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    radius = max(7, round(min(width / 480, height / 270) * 40))
    clean = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(clean, mode="RGB")


def _name_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def add_reveal_name(image: Image.Image, name: str) -> Image.Image:
    """Draw a gold, blue-outlined title where the question mark was, matching the reveal example."""
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    left, right = round(width * 0.57), round(width * 0.97)
    top, bottom = round(height * 0.24), round(height * 0.62)
    max_width, max_height = right - left, bottom - top
    words = (name or "Pokémon").upper().split()

    chosen = None
    for size in range(30, 12, -1):
        font = _name_font(size)
        stroke = max(2, size // 10)
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            box = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke)
            if box[2] - box[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) > 2:
            continue
        text = "\n".join(lines)
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=0,
                                      stroke_width=stroke)
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        if text_width <= max_width and text_height <= max_height:
            chosen = (font, stroke, text, box, text_width, text_height)
            break
    if chosen is None:
        font = ImageFont.load_default(size=13)
        stroke, text = 2, "\n".join(words)
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=0,
                                      stroke_width=stroke)
        chosen = (font, stroke, text, box, box[2] - box[0], box[3] - box[1])

    font, stroke, text, box, text_width, text_height = chosen
    x = round((left + right - text_width) / 2 - box[0])
    y = round((top + bottom - text_height) / 2 - box[1])
    draw.multiline_text((x + 2, y + 2), text, font=font, spacing=0, align="center",
                        fill=(22, 36, 76), stroke_width=stroke + 2,
                        stroke_fill=(22, 36, 76))
    draw.multiline_text((x, y), text, font=font, spacing=0, align="center",
                        fill=(255, 207, 43), stroke_width=stroke,
                        stroke_fill=(25, 78, 164))
    return result


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


def compose(background: Image.Image, artwork: Image.Image, *, silhouette: bool,
            name: str = "") -> Image.Image:
    result = background.convert("RGBA").copy()
    sprite, position = crop_and_fit(artwork, background.size, silhouette=silhouette)
    if not silhouette:
        shadow_alpha = sprite.getchannel("A").filter(ImageFilter.GaussianBlur(2.0))
        shadow_alpha = shadow_alpha.point(lambda value: round(value * 0.42))
        shadow = Image.new("RGBA", sprite.size, (76, 45, 137, 0))
        shadow.putalpha(shadow_alpha)
        result.alpha_composite(shadow, dest=(position[0] + 2, position[1] + 2))
    result.alpha_composite(sprite, dest=position)
    rendered = result.convert("RGB")
    return add_reveal_name(rendered, name) if not silhouette else rendered


def _save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=93, optimize=True, progressive=True, subsampling=0)


def _pokemon_details(form: dict, *, refresh: bool) -> dict:
    name = form["name"]
    url = f"https://pokeapi.co/api/v2/pokemon/{name}/"
    return get_json(url, refresh=refresh)


def _generate_form(form: dict, background: Image.Image, reveal_background: Image.Image,
                   region_dir: Path, *, refresh: bool) -> dict:
    name = form["name"]
    details = _pokemon_details(form, refresh=refresh)
    sprite_url = pokemon_art_url(details)
    if not sprite_url:
        return {
            **form,
            "status": "missing_sprite",
            "sprite_url": None,
            "missing_reason": "PokeAPI provides no artwork or sprite URL for this form.",
        }
    raw = get_bytes(sprite_url, refresh=refresh)
    artwork = Image.open(BytesIO(raw)).convert("RGBA")
    stem = slug(name)
    silhouette_path = region_dir / "silhouettes" / f"{stem}.jpg"
    revealed_path = region_dir / "revealed" / f"{stem}.jpg"
    _save_jpeg(compose(background, artwork, silhouette=True), silhouette_path)
    _save_jpeg(compose(reveal_background, artwork, silhouette=False,
                       name=form["display_name"]), revealed_path)
    return {
        **form,
        "status": "generated",
        "sprite_url": sprite_url,
        "silhouette": silhouette_path.relative_to(ROOT).as_posix(),
        "revealed": revealed_path.relative_to(ROOT).as_posix(),
    }


def list_forms(region: str, *, workers: int, refresh: bool) -> tuple[list[dict], list[dict]]:
    pokedex_names = REGION_POKEDEXES.get(region)
    if pokedex_names is None:
        raise ValueError(f"Region {region!r} is not configured; choose one of {REGIONS}.")
    pokedexes = [
        get_json(f"https://pokeapi.co/api/v2/pokedex/{name}/", refresh=refresh)
        for name in pokedex_names
    ]

    # Merge split/DLC dexes without counting the same species more than once. If a species
    # appears in more than one dex, retain its first regional entry as the canonical number.
    unique_entries = {}
    for dex_order, pokedex in enumerate(pokedexes):
        for entry in sorted(pokedex["pokemon_entries"], key=lambda row: row["entry_number"]):
            species = entry["pokemon_species"]
            unique_entries.setdefault(species["name"], (
                dex_order, pokedex["name"], entry["entry_number"], species["url"]))
    ordered_entries = sorted(
        ((species_name, *details) for species_name, details in unique_entries.items()),
        key=lambda row: (row[1], row[3], row[0]))

    def fetch_species(entry: tuple[str, int, str, int, str]) -> tuple[int, str, str, dict]:
        species_name, _dex_order, pokedex_name, dex_number, species_url = entry
        record = get_json(species_url, refresh=refresh)
        return dex_number, pokedex_name, species_name, record

    species_data = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_species, entry) for entry in ordered_entries]
        for future in as_completed(futures):
            species_data.append(future.result())
    species_data.sort(key=lambda item: (pokedex_names.index(item[1]), item[0], item[2]))

    forms = []
    seen = set()
    for dex_number, pokedex_name, species_name, record in species_data:
        species_label = _english_name(record, species_name)
        for variety in record.get("varieties", []):
            pokemon = variety["pokemon"]
            pokemon_name = pokemon["name"]
            if pokemon_name in seen:
                continue
            seen.add(pokemon_name)
            forms.append({
                "dex_number": dex_number,
                "source_pokedex": pokedex_name,
                "species": species_name,
                "species_label": species_label,
                "name": pokemon_name,
                "display_name": pokemon_display_name(species_name, pokemon_name, species_label),
                "is_default": bool(variety.get("is_default")),
            })
    return pokedexes, forms


def generate(region: str, template_path: Path, *, workers: int = 8,
             limit: int | None = None, refresh: bool = False) -> dict:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    region_dir = ROOT / "regions" / region
    image = Image.open(template_path).convert("RGB")
    background = clean_background(image)
    reveal_background = remove_question_mark(background)
    pokedexes, forms = list_forms(region, workers=workers, refresh=refresh)
    species_count = len({form["species"] for form in forms})
    dex_entry_count = sum(len(pokedex["pokemon_entries"]) for pokedex in pokedexes)
    if limit is not None:
        forms = forms[:max(0, limit)]
    dex_label = " + ".join(pokedex["name"] for pokedex in pokedexes)
    print(f"[{region}] PokeAPI {dex_label}: {dex_entry_count} source entries, "
          f"{species_count} unique species, {len(forms)} distinct forms with {workers} workers.",
          flush=True)

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate_form, form, background, reveal_background,
                                  region_dir, refresh=refresh): form for form in forms}
        for future in as_completed(futures):
            form = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # preserve partial progress and explain any PokeAPI failures
                result = {**form, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            completed += 1
            if completed % 25 == 0 or completed == len(forms):
                print(f"[{region}] Rendered {completed}/{len(forms)} forms", flush=True)

    results.sort(key=lambda row: (row["dex_number"], row["name"]))
    pokedex_sources = [
        {
            "name": pokedex["name"],
            "url": f"https://pokeapi.co/api/v2/pokedex/{pokedex['name']}/",
            "entry_count": len(pokedex["pokemon_entries"]),
        }
        for pokedex in pokedexes
    ]
    manifest = {
        "region": region,
        "pokedex": pokedexes[0]["name"],
        "pokedex_url": pokedex_sources[0]["url"],
        "pokedexes": pokedex_sources,
        "pokedex_entry_count": dex_entry_count,
        "species_count": species_count,
        "form_count": len(forms),
        "generated_count": sum(row["status"] == "generated" for row in results),
        "missing_or_failed_count": sum(row["status"] != "generated" for row in results),
        "form_policy": "all PokeAPI varieties for every unique species across the configured regional Pokédexes",
        "results": results,
    }
    region_dir.mkdir(parents=True, exist_ok=True)
    (region_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_summary = ", ".join(
        f"[`{source['name']}`]({source['url']}) ({source['entry_count']} entries)"
        for source in pokedex_sources)
    (region_dir / "README.md").write_text(
        f"# {region.title()} Pokémon quiz images\n\n"
        f"Source Pokédexes: {source_summary}. There are {species_count} unique species "
        f"across {dex_entry_count} regional entries.\n\n"
        f"The generator expands every listed species to all PokeAPI varieties. It rendered "
        f"{manifest['generated_count']} forms as paired images in `silhouettes/` and `revealed/`. "
        f"Each revealed image replaces the question mark with the Pokémon/form name. This includes "
        f"regional forms and battle forms such as Ultra Necrozma when PokeAPI provides official "
        f"artwork. See `manifest.json` for each file and source URL.\n\n"
        "Pokémon names and artwork are the property of their respective rights holders; artwork "
        "is fetched from PokeAPI's sprite data.\n",
        encoding="utf-8")
    return manifest


def generate_all(template_path: Path, *, workers: int = 8,
                 limit: int | None = None, refresh: bool = False) -> dict[str, dict]:
    """Build every configured region concurrently within a shared worker budget."""
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    parallel_regions = min(len(REGIONS), workers)
    base_workers, extra_workers = divmod(workers, parallel_regions)
    allocations = {
        region: base_workers + (1 if index < extra_workers else 0)
        for index, region in enumerate(REGIONS)
    }
    print(f"Generating {len(REGIONS)} regions with up to {parallel_regions} regions "
          f"in parallel ({workers} total workers).", flush=True)

    manifests = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=parallel_regions) as executor:
        futures = {
            executor.submit(generate, region, template_path,
                            workers=allocations[region], limit=limit, refresh=refresh): region
            for region in REGIONS
        }
        for future in as_completed(futures):
            region = futures[future]
            try:
                manifests[region] = future.result()
                manifest = manifests[region]
                print(f"[{region}] complete: {manifest['generated_count']}/"
                      f"{manifest['form_count']} forms; "
                      f"{manifest['missing_or_failed_count']} missing/failed.", flush=True)
            except Exception as exc:
                errors[region] = f"{type(exc).__name__}: {exc}"
                print(f"[{region}] region job failed: {errors[region]}", flush=True)
    if errors:
        details = "; ".join(f"{region}: {error}" for region, error in errors.items())
        raise RuntimeError(f"One or more region jobs failed: {details}")
    return {region: manifests[region] for region in REGIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="alola", choices=("all", *REGIONS),
                        help="regional collection to process, or all regions in parallel")
    parser.add_argument("--template", type=Path,
                        default=ROOT / "assets" / "quiz-template.jpg",
                        help="mystery-quiz image to reuse as the background")
    parser.add_argument("--workers", type=int, default=8,
                        help="worker budget; global across regions for --region all (default: 8)")
    parser.add_argument("--limit", type=int,
                        help="optional maximum number of forms per region, useful for a quick test")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore HTTP cache and refetch PokeAPI metadata and sprites")
    args = parser.parse_args()
    if args.region == "all":
        manifests = generate_all(args.template, workers=args.workers,
                                 limit=args.limit, refresh=args.refresh)
        total_forms = sum(manifest["form_count"] for manifest in manifests.values())
        generated = sum(manifest["generated_count"] for manifest in manifests.values())
        failed = sum(manifest["missing_or_failed_count"] for manifest in manifests.values())
        print(f"All {len(manifests)} regions complete: {generated}/{total_forms} forms generated; "
              f"{failed} missing/failed. See the regions/ manifests.", flush=True)
    else:
        manifest = generate(args.region, args.template, workers=args.workers,
                            limit=args.limit, refresh=args.refresh)
        print(f"Done: {manifest['generated_count']} generated; "
              f"{manifest['missing_or_failed_count']} missing/failed. "
              f"See regions/{args.region}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
