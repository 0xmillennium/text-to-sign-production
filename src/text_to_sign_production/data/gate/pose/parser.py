"""OpenPose frame parsing into normalized pose truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from text_to_sign_production.data.gate.pose.schema import (
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    EXPECTED_OPENPOSE_PERSON_KEYS,
    EXPECTED_OPENPOSE_TOP_LEVEL_KEYS,
    OPENPOSE_CHANNEL_SPECS,
)
from text_to_sign_production.data.gate.pose.types import (
    FloatArray,
    ParsedFrame,
    ParsedPerson,
    ParsedPoseChannel,
    PoseChannel,
    PoseDiagnostic,
    PoseDiagnosticCode,
    PoseDiagnosticSeverity,
)


def _diagnostic(
    code: PoseDiagnosticCode,
    message: str,
    *,
    frame_index: int | None = None,
    person_index: int | None = None,
    channel: PoseChannel | None = None,
    severity: PoseDiagnosticSeverity = PoseDiagnosticSeverity.ERROR,
) -> PoseDiagnostic:
    return PoseDiagnostic(
        code=code,
        severity=severity,
        message=message,
        frame_index=frame_index,
        person_index=person_index,
        channel=channel,
    )


def zero_channel(channel: PoseChannel) -> ParsedPoseChannel:
    """Return a zero-filled parsed channel in normalized image coordinates."""
    _, point_count = OPENPOSE_CHANNEL_SPECS[channel]
    return ParsedPoseChannel(
        channel=channel,
        coordinates=np.zeros((point_count, 2), dtype=np.float32),
        confidences=np.zeros((point_count,), dtype=np.float32),
    )


def _reshape_flat_keypoints(
    flat_values: list[object],
    expected_points: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if len(flat_values) != expected_points * 3:
        raise ValueError(f"Expected {expected_points * 3} values, got {len(flat_values)}.")
    raw = np.asarray(flat_values, dtype=np.float32).reshape(expected_points, 3)
    return raw, raw[:, :2].copy(), raw[:, 2].copy()


def parse_frame_payload(
    payload: object,
    *,
    frame_index: int,
    source_path: Path | None = None,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> ParsedFrame:
    """Parse one OpenPose payload into normalized image coordinate truth.

    The returned coordinates are always normalized image coordinates:
    x is divided by ``canvas_width`` and y is divided by ``canvas_height``.
    They are not pixel coordinates.
    """
    diagnostics: list[PoseDiagnostic] = []
    if not isinstance(payload, dict):
        return ParsedFrame(
            frame_index=frame_index,
            source_path=source_path,
            people=(),
            frame_valid=False,
            diagnostics=(
                _diagnostic(
                    PoseDiagnosticCode.TOP_LEVEL_NOT_OBJECT,
                    "OpenPose frame payload must be an object.",
                    frame_index=frame_index,
                ),
            ),
        )

    for key in sorted(EXPECTED_OPENPOSE_TOP_LEVEL_KEYS.difference(payload.keys())):
        diagnostics.append(
            _diagnostic(
                PoseDiagnosticCode.MISSING_TOP_LEVEL_KEY,
                f"OpenPose frame missing top-level key {key!r}.",
                frame_index=frame_index,
                severity=PoseDiagnosticSeverity.WARNING,
            )
        )

    people_payload = payload.get("people")
    if not isinstance(people_payload, list):
        return ParsedFrame(
            frame_index=frame_index,
            source_path=source_path,
            people=(),
            frame_valid=False,
            diagnostics=tuple(
                diagnostics
                + [
                    _diagnostic(
                        PoseDiagnosticCode.PEOPLE_NOT_LIST,
                        "OpenPose people field must be a list.",
                        frame_index=frame_index,
                    )
                ]
            ),
        )
    if not people_payload:
        return ParsedFrame(
            frame_index=frame_index,
            source_path=source_path,
            people=(),
            frame_valid=False,
            diagnostics=tuple(
                diagnostics
                + [
                    _diagnostic(
                        PoseDiagnosticCode.PEOPLE_EMPTY,
                        "OpenPose people field is empty.",
                        frame_index=frame_index,
                        severity=PoseDiagnosticSeverity.WARNING,
                    )
                ]
            ),
        )

    parsed_people: list[ParsedPerson] = []
    for person_index, person_payload in enumerate(people_payload):
        parsed_people.append(
            _parse_person_payload(
                person_payload,
                frame_index=frame_index,
                person_index=person_index,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
        )

    return ParsedFrame(
        frame_index=frame_index,
        source_path=source_path,
        people=tuple(parsed_people),
        frame_valid=any(person.person_valid for person in parsed_people),
        diagnostics=tuple(diagnostics),
    )


def _parse_person_payload(
    payload: object,
    *,
    frame_index: int,
    person_index: int,
    canvas_width: int,
    canvas_height: int,
) -> ParsedPerson:
    if not isinstance(payload, dict):
        return ParsedPerson(
            channels={channel: zero_channel(channel) for channel in OPENPOSE_CHANNEL_SPECS},
            person_valid=False,
            face_missing=True,
            out_of_bounds_coordinate_count=0,
            has_any_zeroed_landmark=False,
            diagnostics=(
                _diagnostic(
                    PoseDiagnosticCode.PERSON_NOT_OBJECT,
                    "OpenPose person entry must be an object.",
                    frame_index=frame_index,
                    person_index=person_index,
                ),
            ),
        )

    diagnostics: list[PoseDiagnostic] = []
    for key in sorted(EXPECTED_OPENPOSE_PERSON_KEYS.difference(payload.keys())):
        diagnostics.append(
            _diagnostic(
                PoseDiagnosticCode.MISSING_PERSON_KEY,
                f"OpenPose person missing key {key!r}.",
                frame_index=frame_index,
                person_index=person_index,
                severity=PoseDiagnosticSeverity.WARNING,
            )
        )

    person_valid = True
    face_missing = False
    out_of_bounds = 0
    has_zeroed = False
    channels: dict[PoseChannel, ParsedPoseChannel] = {}

    for channel, (raw_key, expected_points) in OPENPOSE_CHANNEL_SPECS.items():
        raw_values = payload.get(raw_key, [])
        if not isinstance(raw_values, list):
            diagnostics.append(
                _diagnostic(
                    PoseDiagnosticCode.CHANNEL_NOT_LIST,
                    f"OpenPose channel {raw_key!r} must be a list.",
                    frame_index=frame_index,
                    person_index=person_index,
                    channel=channel,
                )
            )
            person_valid = False
            channels[channel] = zero_channel(channel)
            continue
        if len(raw_values) != expected_points * 3:
            diagnostics.append(
                _diagnostic(
                    PoseDiagnosticCode.UNEXPECTED_CHANNEL_LENGTH,
                    f"OpenPose channel {raw_key!r} has unexpected length {len(raw_values)}.",
                    frame_index=frame_index,
                    person_index=person_index,
                    channel=channel,
                )
            )
            person_valid = False
            channels[channel] = zero_channel(channel)
            continue
        try:
            raw_array, coordinates, confidences = _reshape_flat_keypoints(
                raw_values, expected_points
            )
        except (TypeError, ValueError):
            diagnostics.append(
                _diagnostic(
                    PoseDiagnosticCode.CHANNEL_NON_NUMERIC,
                    f"OpenPose channel {raw_key!r} contains non-numeric values.",
                    frame_index=frame_index,
                    person_index=person_index,
                    channel=channel,
                )
            )
            person_valid = False
            channels[channel] = zero_channel(channel)
            continue

        coordinates[:, 0] = coordinates[:, 0] / np.float32(canvas_width)
        coordinates[:, 1] = coordinates[:, 1] / np.float32(canvas_height)

        x_values = raw_array[:, 0]
        y_values = raw_array[:, 1]
        channel_out_of_bounds = int(np.count_nonzero((x_values < 0) | (x_values >= canvas_width)))
        channel_out_of_bounds += int(np.count_nonzero((y_values < 0) | (y_values >= canvas_height)))
        out_of_bounds += channel_out_of_bounds
        if channel_out_of_bounds:
            diagnostics.append(
                _diagnostic(
                    PoseDiagnosticCode.OUT_OF_BOUNDS_COORDINATE,
                    f"{channel.value} has {channel_out_of_bounds} out-of-bounds coordinates.",
                    frame_index=frame_index,
                    person_index=person_index,
                    channel=channel,
                    severity=PoseDiagnosticSeverity.WARNING,
                )
            )

        if bool(np.any(np.all(raw_array == 0.0, axis=1))):
            has_zeroed = True
            diagnostics.append(
                _diagnostic(
                    PoseDiagnosticCode.ZEROED_LANDMARK,
                    f"{channel.value} has at least one zeroed landmark.",
                    frame_index=frame_index,
                    person_index=person_index,
                    channel=channel,
                    severity=PoseDiagnosticSeverity.WARNING,
                )
            )

        if channel is PoseChannel.FACE and float(np.sum(confidences)) <= 0.0:
            face_missing = True
            diagnostics.append(
                _diagnostic(
                    PoseDiagnosticCode.FACE_MISSING,
                    "Face channel has no positive confidence.",
                    frame_index=frame_index,
                    person_index=person_index,
                    channel=channel,
                    severity=PoseDiagnosticSeverity.WARNING,
                )
            )

        channels[channel] = ParsedPoseChannel(
            channel=channel,
            coordinates=coordinates,
            confidences=confidences,
        )

    return ParsedPerson(
        channels=channels,
        person_valid=person_valid,
        face_missing=face_missing,
        out_of_bounds_coordinate_count=out_of_bounds,
        has_any_zeroed_landmark=has_zeroed,
        diagnostics=tuple(diagnostics),
    )


def parse_frame_file(
    path: Path,
    *,
    frame_index: int,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> ParsedFrame:
    """Read and parse one OpenPose frame JSON file."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
    except json.JSONDecodeError:
        return ParsedFrame(
            frame_index=frame_index,
            source_path=path,
            people=(),
            frame_valid=False,
            diagnostics=(
                _diagnostic(
                    PoseDiagnosticCode.JSON_DECODE_ERROR,
                    "Frame JSON could not be decoded.",
                    frame_index=frame_index,
                ),
            ),
        )
    except UnicodeDecodeError:
        return ParsedFrame(
            frame_index=frame_index,
            source_path=path,
            people=(),
            frame_valid=False,
            diagnostics=(
                _diagnostic(
                    PoseDiagnosticCode.UNICODE_DECODE_ERROR,
                    "Frame JSON could not be decoded as UTF-8.",
                    frame_index=frame_index,
                ),
            ),
        )
    except OSError as exc:
        return ParsedFrame(
            frame_index=frame_index,
            source_path=path,
            people=(),
            frame_valid=False,
            diagnostics=(
                _diagnostic(
                    PoseDiagnosticCode.FRAME_READ_ERROR,
                    f"Frame JSON could not be read: {exc.__class__.__name__}.",
                    frame_index=frame_index,
                ),
            ),
        )
    return parse_frame_payload(
        payload,
        frame_index=frame_index,
        source_path=path,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


__all__ = ["parse_frame_file", "parse_frame_payload", "zero_channel"]
