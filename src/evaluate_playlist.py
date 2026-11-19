"""Evaluate the guarded playlist on representative catalog and input cases."""

from .playlist import CATALOG, MIN_SIMILARITY, build_playlist
from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs(CATALOG)
    pop = {"genre": "pop", "mood": "happy", "energy": 0.8}
    lofi = {"genre": "lofi", "mood": "chill", "energy": 0.4}
    unsupported = {"genre": "opera", "mood": "spooky", "energy": 0.8}
    full = build_playlist(songs, **lofi, limit=2, diversity=True)
    partial = build_playlist(songs, **pop)
    excluded = build_playlist(songs, **pop, excluded_artists=(" neon ECHO ",))
    empty = build_playlist([], **pop)
    uncovered = build_playlist(songs, **unsupported)
    unguarded = recommend_songs(unsupported, songs, k=3)

    try:
        build_playlist(songs, **{**pop, "energy": float("nan")})
    except ValueError as error:
        rejects_invalid_energy = str(error) == "energy must be between 0 and 1"
    else:
        rejects_invalid_energy = False

    cases = {
        "Lo-fi produces two different artists with diversity enabled": (
            full.status == "complete" and len({track[0]["artist"] for track in full.tracks}) == 2
        ),
        "Pop returns one strong track and does not fill two remaining slots": (
            partial.status == "partial" and [track[0]["id"] for track in partial.tracks] == [1]
        ),
        "Case-insensitive artist exclusion removes the only eligible pop track": (
            excluded.status == "abstain" and excluded.blocked_artist_count == 2
        ),
        "An empty catalog returns an empty playlist": empty.status == "abstain",
        "Unsupported request abstains when the original ranks three weak tracks": (
            uncovered.status == "abstain" and len(unguarded) == 3
            and all(score < MIN_SIMILARITY for _, score, _ in unguarded)
        ),
        "Non-finite energy is rejected": rejects_invalid_energy,
    }
    for name, passed in cases.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n{sum(cases.values())}/{len(cases)} checks passed")
    if not all(cases.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
