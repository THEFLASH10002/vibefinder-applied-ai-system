"""Build a playlist that can return fewer tracks when the catalog is a poor fit."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from .recommender import SCORING_STRATEGIES, load_songs, recommend_songs

CATALOG = Path(__file__).resolve().parents[1] / "data" / "songs.csv"
MIN_SIMILARITY = 0.65


@dataclass(frozen=True)
class PlaylistResult:
    tracks: list[tuple[dict, float, str]]
    blocked_artist_count: int
    weak_match_count: int
    requested_count: int

    @property
    def status(self) -> str:
        if not self.tracks:
            return "abstain"
        if len(self.tracks) < self.requested_count:
            return "partial"
        return "complete"


def build_playlist(
    songs: list[dict], *, genre: str, mood: str, energy: float,
    limit: int = 3, excluded_artists: tuple[str, ...] = (),
    recipe: str = "baseline", diversity: bool = False,
) -> PlaylistResult:
    """Screen candidates before applying the existing optional diversity reranker."""
    genre, mood = genre.strip().casefold(), mood.strip().casefold()
    if not genre or not mood:
        raise ValueError("genre and mood must be nonempty")
    if not 0 <= energy <= 1:
        raise ValueError("energy must be between 0 and 1")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    if recipe not in SCORING_STRATEGIES:
        raise ValueError(f"Unknown scoring recipe: {recipe}")
    blocked = {artist.strip().casefold() for artist in excluded_artists}
    if "" in blocked:
        raise ValueError("excluded artist names must be nonempty")
    prefs = {"genre": genre, "mood": mood, "energy": energy}
    available = [song for song in songs if song["artist"].strip().casefold() not in blocked]
    ranked = recommend_songs(prefs, available, k=len(available), recipe=recipe)
    eligible = [entry for entry in ranked if entry[1] >= MIN_SIMILARITY]
    tracks = eligible[:limit]
    if diversity:
        tracks = recommend_songs(
            prefs, [song for song, _, _ in eligible], k=limit, recipe=recipe, diversity=True
        )
    return PlaylistResult(tracks, len(songs) - len(available), len(ranked) - len(eligible), limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genre", required=True)
    parser.add_argument("--mood", required=True)
    parser.add_argument("--energy", type=float, required=True, help="target energy from 0 to 1")
    parser.add_argument("--limit", type=int, default=3, help="maximum tracks, from 1 to 20")
    parser.add_argument("--exclude-artist", action="append", default=[], help="repeat to exclude multiple artists")
    parser.add_argument("--mode", choices=tuple(SCORING_STRATEGIES), default="baseline")
    parser.add_argument("--diversity", action="store_true")
    args = parser.parse_args()
    try:
        result = build_playlist(
            load_songs(CATALOG), genre=args.genre, mood=args.mood, energy=args.energy,
            limit=args.limit, excluded_artists=tuple(args.exclude_artist),
            recipe=args.mode, diversity=args.diversity,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Status: {result.status} ({len(result.tracks)}/{result.requested_count} tracks)")
    print(f"Screened out: {result.blocked_artist_count} excluded artist track(s), "
          f"{result.weak_match_count} track(s) below {MIN_SIMILARITY:.2f} similarity")
    if not result.tracks:
        print("No tracks meet the request. Try a different genre, mood, or energy target.")
    for rank, (song, score, explanation) in enumerate(result.tracks, 1):
        print(f"{rank}. {song['title']} — {song['artist']} ({score:.3f})")
        print(f"   {explanation}")


if __name__ == "__main__":
    main()
