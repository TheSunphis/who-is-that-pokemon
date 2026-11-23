# Who Is That Pokémon?

A region-organized collection of Pokémon quiz cards. Each form gets a pair of images using the same
background: a black silhouette and the revealed PokeAPI official-artwork sprite. Revealed cards replace
the question mark with the Pokémon/form name in gold text with a blue outline.

## Current build

- **Alola:** generated from PokeAPI's `updated-alola` Pokédex.
- **Forms:** every PokeAPI variety for each Alola Pokédex species is included when artwork exists,
  including Alolan regional forms and forms such as Ultra Necrozma.
- **Image generation:** parallelized with a configurable worker count; the local HTTP cache prevents
  repeat downloads. The output manifest records the exact species, form and sprite URL.

## Generate Alola

Install the Python dependencies:

```bash
python3 -m pip install pillow opencv-python numpy
```

Place the mystery-quiz template at `assets/quiz-template.jpg`, then run:

```bash
python3 scripts/generate_region.py --region alola --workers 8
```

The script fetches the official artwork URL from PokeAPI, caches it and creates:

- `regions/alola/silhouettes/<pokemon-form>.jpg`
- `regions/alola/revealed/<pokemon-form>.jpg`
- `regions/alola/manifest.json`

For a small smoke test, add `--limit 10`. To ignore the cache and fetch fresh data, add `--refresh`.

## Regions

The repository is laid out for Kanto, Johto, Hoenn, Sinnoh, Unova, Kalos, Alola, Galar, Hisui and
Paldea under `regions/`. Alola is the first generated region; the remaining folders are ready for
later builds.

## Data and artwork

Pokémon data and official-artwork image URLs are retrieved from [PokeAPI](https://pokeapi.co/).
Pokémon names, characters, logos and artwork belong to their respective rights holders. This is an
unofficial fan project; review applicable rights and platform terms before publishing the generated
art publicly.
