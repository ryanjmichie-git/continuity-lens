from __future__ import annotations

from continuity_lens.dataset import (
    build_transition_records,
    find_davis_root,
    validate_split_isolation,
)


def test_deterministic_cases_and_split_isolation(tiny_davis) -> None:
    davis = find_davis_root(tiny_davis)
    first = build_transition_records(davis, split="dev", horizon=4)
    second = build_transition_records(davis, split="dev", horizon=4)
    test = build_transition_records(davis, split="test", horizon=4)
    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]
    assert {record.corruption for record in first} == {
        "continuous",
        "temporal_skip",
        "block_reorder",
        "cross_video_splice",
    }
    assert all(len(record.context_paths) == 12 for record in first)
    assert all(len(record.target_paths) == 4 for record in first)
    assert all(len(set(record.context_paths + record.target_paths)) == 16 for record in first[:3])
    validate_split_isolation(first, test)
