# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

from dynworkflow.get_walltime_and_ranks_aggregated_job import parse_timestamp


def test_parse_timestamp(tmp_path, monkeypatch):
    # Using monkeypatch to safely handle directory changes
    monkeypatch.chdir(tmp_path)

    # 1. Test the "New" format (ISO-like)
    ts_str_new = "2026-04-21 11:24:53.758"
    result_new = parse_timestamp(ts_str_new)

    assert result_new is not None, "Failed to parse new timestamp format"
    assert result_new.year == 2026
    assert result_new.microsecond == 758000

    # 2. Test the "Old" format (ctime-like)
    # Note: This format lacks a year, so Python defaults to 1900
    ts_str_old = "Tue Apr 21 11:24:53"
    result_old = parse_timestamp(ts_str_old)

    assert result_old is not None, "Failed to parse old timestamp format"
    assert result_old.month == 4
    assert result_old.hour == 11

    # 3. Test invalid input
    assert parse_timestamp("not-a-date") is None
