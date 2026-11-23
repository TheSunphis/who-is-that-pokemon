# Who Is That Pokémon?

A region-organized collection of Pokémon quiz cards. Each Pokémon/form gets a pair of cards using the
same background: a black silhouette and the revealed PokeAPI official-artwork sprite. Silhouette
cards keep the question mark; revealed cards replace it with the Pokémon/form name in gold with a
blue outline.

## Regional coverage

The repository contains generated builds for all ten regions: Kanto, Johto, Hoenn, Sinnoh, Unova,
Kalos, Alola, Galar, Hisui, and Paldea. The generator uses PokeAPI's current regional Pokédexes,
combining split/DLC dexes where
applicable (Kalos, Galar, and Paldea; plus Let's Go Kanto). Every listed species expands to its
PokeAPI varieties when artwork is available, including regional and battle forms such as Ultra
Necrozma. Forms without a PokeAPI sprite URL remain in the manifest as `missing_sprite`; the
renderer does not substitute artwork from a different form. See each `regions/<region>/manifest.json`
for source dexes, forms, and artwork URLs.

## Generate

Install the Python dependencies:

```bash
python3 -m pip install pillow opencv-python numpy
```

Place the quiz template at `assets/quiz-template.jpg`. Generate one region:

```bash
python3 scripts/generate_region.py --region alola --workers 8
```

Generate all ten regions concurrently (the worker count is a global budget shared across them):

```bash
python3 scripts/generate_region.py --region all --workers 20
```

Each region writes paired files and a manifest under `regions/<region>/`:

- `silhouettes/<pokemon-form>.jpg`
- `revealed/<pokemon-form>.jpg`
- `manifest.json`

For a smoke test, add `--limit 10` (for `--region all`, this limits each region). The local HTTP
cache under `.cache/` avoids repeat downloads; add `--refresh` to refetch PokeAPI data and artwork.

## Data and artwork

Pokémon data and official-artwork image URLs are retrieved from [PokeAPI](https://pokeapi.co/).
Pokémon names, characters, logos and artwork belong to their respective rights holders. This is an
unofficial fan project; review applicable rights and platform terms before publishing the generated
art publicly.
