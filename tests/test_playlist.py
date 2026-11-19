from dataclasses import replace

import pytest

from src.playlist import CATALOG, MIN_SIMILARITY, build_playlist
from src.recommender import SCORING_STRATEGIES, load_songs


@pytest.mark.parametrize("energy", [-0.01, 1.01, float("inf"), float("nan")])
def test_invalid_energy_is_rejected_even_with_empty_catalog(energy):
    with pytest.raises(ValueError, match="energy must be between 0 and 1"):
        build_playlist([], genre="pop", mood="happy", energy=energy)


def test_gate_includes_exact_threshold(monkeypatch):
    strategy = replace(SCORING_STRATEGIES["baseline"], genre=MIN_SIMILARITY,
                       mood=1 - MIN_SIMILARITY, energy=0)
    monkeypatch.setitem(SCORING_STRATEGIES, "threshold-test", strategy)
    songs = [dict(id=1, genre="pop", mood="calm", energy=0.5, artist="Example")]
    result = build_playlist(songs, genre=" POP ", mood="happy", energy=0.5,
                            recipe="threshold-test")
    assert result.status == "partial"
    assert result.tracks[0][1] == MIN_SIMILARITY


def test_diversity_does_not_reintroduce_excluded_or_weak_tracks():
    songs = load_songs(CATALOG)
    result = build_playlist(songs, genre="lofi", mood="chill", energy=0.4,
                            limit=10, excluded_artists=(" loroom ",), diversity=True)
    assert result.status == "partial"
    assert [song["id"] for song, _, _ in result.tracks] == [4]
    assert all(song["artist"].casefold() != "loroom" for song, _, _ in result.tracks)
