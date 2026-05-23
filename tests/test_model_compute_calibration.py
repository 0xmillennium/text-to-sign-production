from text_to_sign_production.workflows.model.processing.calibration import (
    benchmark_compute_candidates,
    select_compute_calibration_overrides,
)


def test_highest_throughput_selected() -> None:
    selected, flags = select_compute_calibration_overrides(
        [
            {"parameter": "tokenizer_batch_size", "value": 128, "units_per_second": 100.0, "peak_cuda_memory_reserved_fraction": 0.1},
            {"parameter": "tokenizer_batch_size", "value": 256, "units_per_second": 140.0, "peak_cuda_memory_reserved_fraction": 0.2},
        ]
    )
    assert selected["tokenizer_batch_size"] == 256
    assert flags["cpu_or_io_bound"] == []


def test_near_tie_picks_smaller_batch() -> None:
    selected, _flags = select_compute_calibration_overrides(
        [
            {"parameter": "decode_batch_size", "value": 128, "units_per_second": 100.0, "peak_cuda_memory_reserved_fraction": 0.1},
            {"parameter": "decode_batch_size", "value": 256, "units_per_second": 104.0, "peak_cuda_memory_reserved_fraction": 0.2},
        ]
    )
    assert selected["decode_batch_size"] == 128


def test_oom_candidate_skipped() -> None:
    selected, _flags = select_compute_calibration_overrides(
        [
            {"parameter": "text_to_token_batch_size", "value": 128, "units_per_second": 90.0, "peak_cuda_memory_reserved_fraction": 0.2},
            {"parameter": "text_to_token_batch_size", "value": 512, "units_per_second": 200.0, "peak_cuda_memory_reserved_fraction": 0.4, "oom": True},
        ]
    )
    assert selected["text_to_token_batch_size"] == 128


def test_memory_cap_respected() -> None:
    selected, _flags = select_compute_calibration_overrides(
        [
            {"parameter": "reconstruction_batch_size", "value": 128, "units_per_second": 90.0, "peak_cuda_memory_reserved_fraction": 0.2},
            {"parameter": "reconstruction_batch_size", "value": 512, "units_per_second": 200.0, "peak_cuda_memory_reserved_fraction": 0.9},
        ]
    )
    assert selected["reconstruction_batch_size"] == 128


def test_cpu_or_io_bound_flag_emitted_when_low_memory_and_flat_throughput() -> None:
    selected, flags = select_compute_calibration_overrides(
        [
            {"parameter": "tokenizer_batch_size", "value": 128, "units_per_second": 100.0, "peak_cuda_memory_reserved_fraction": 0.01},
            {"parameter": "tokenizer_batch_size", "value": 256, "units_per_second": 102.0, "peak_cuda_memory_reserved_fraction": 0.02},
        ]
    )
    assert selected["tokenizer_batch_size"] == 128
    assert flags["cpu_or_io_bound"] == ["tokenizer_batch_size"]


def test_benchmark_compute_candidates_runs_synthetic_candidate_loop() -> None:
    rows = benchmark_compute_candidates(
        {"tokenizer_batch_size": [2, 4]},
        max_batches_per_candidate=2,
    )

    assert [row["value"] for row in rows] == [2, 4]
    assert all(row["units_processed"] in {4, 8} for row in rows)
    assert all(isinstance(row["units_per_second"], float) for row in rows)
    assert sum(1 for row in rows if row["selected"]) == 1
