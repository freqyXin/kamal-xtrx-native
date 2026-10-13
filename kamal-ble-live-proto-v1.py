#!/usr/bin/env python3

import argparse
import importlib.util
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np


def load_decoder(path):
    spec = importlib.util.spec_from_file_location(
        "kamal_ble_decode_proto_v2",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load decoder: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def read_exact_chunk(stream, wanted):
    buf = bytearray(wanted)
    view = memoryview(buf)
    offset = 0

    while offset < wanted:
        count = stream.readinto(view[offset:])

        if not count:
            break

        offset += count

    return bytes(view[:offset])


def reader_worker(stream, output_queue, chunk_bytes, errors):
    sample_cursor = 0

    try:
        while True:
            data = read_exact_chunk(stream, chunk_bytes)

            if not data:
                break

            if len(data) % 4:
                raise RuntimeError(
                    f"stream ended on partial CS16 sample: {len(data)} bytes"
                )

            output_queue.put((sample_cursor, data))

            sample_cursor += len(data) // 4

            if len(data) != chunk_bytes:
                break

    except BaseException as exc:
        errors.append(exc)

    finally:
        output_queue.put(None)


def analyze_chunk(
    decoder,
    raw,
    base_sample,
    rate,
    channel,
    sps,
    power_window_us,
    burst_db,
    merge_us,
    pre_us,
    post_us,
    min_corr,
    min_dev_khz,
    max_dev_khz,
    max_sync_errors,
):
    total_complex = len(raw) // 2

    win_samples = int(
        round(power_window_us * 1e-6 * rate)
    )

    powers = decoder.measure_window_power(
        raw,
        total_complex,
        win_samples,
    )

    if len(powers) == 0:
        return [], 0, 0

    median_power = float(np.median(powers))

    threshold_power = (
        median_power
        * (10.0 ** (burst_db / 10.0))
    )

    burst_indexes = np.flatnonzero(
        powers >= threshold_power
    )

    if len(burst_indexes) == 0:
        return [], 0, 0

    merge_windows = max(
        1,
        int(round(
            merge_us / power_window_us
        )),
    )

    runs = decoder.merge_window_indexes(
        burst_indexes,
        merge_windows,
    )

    pre_samples = int(
        round(pre_us * 1e-6 * rate)
    )

    post_samples = int(
        round(post_us * 1e-6 * rate)
    )

    regions = []

    for first_window, last_window in runs:
        start = max(
            0,
            first_window * win_samples
            - pre_samples,
        )

        end = min(
            total_complex,
            (last_window + 1) * win_samples
            + post_samples,
        )

        regions.append((start, end))

    regions = decoder.merge_sample_regions(
        regions
    )

    packets = []
    candidate_count = 0

    for start, end in regions:
        found, diagnostics = decoder.decode_region(
            raw=raw,
            region_start=start,
            region_end=end,
            rate=rate,
            sps=sps,
            channel=channel,
            min_corr=min_corr,
            min_dev_hz=min_dev_khz * 1000.0,
            max_dev_hz=max_dev_khz * 1000.0,
            max_sync_errors=max_sync_errors,
        )

        candidate_count += len(diagnostics)

        for packet in found:
            packet = dict(packet)

            packet["time_s"] += (
                base_sample / rate
            )

            packets.append(packet)

    return (
        decoder.dedupe_packets(packets),
        len(regions),
        candidate_count,
    )


def packet_is_duplicate(packet, recent):
    for previous_time, previous_pdu in recent:
        if previous_pdu != packet["pdu"]:
            continue

        if abs(
            previous_time - packet["time_s"]
        ) < 20e-6:
            return True

    return False


def print_packet(decoder, packet, channel):
    pdu_name = decoder.PDU_NAMES.get(
        packet["pdu_type"],
        f"TYPE_{packet['pdu_type']}",
    )

    fields = packet["fields"]

    parts = [
        f"t={packet['time_s']:.6f}s",
        f"CH{channel}",
        pdu_name,
    ]

    for key in (
        "AdvA",
        "ScanA",
        "InitA",
        "TargetA",
    ):
        if key in fields:
            parts.append(
                f"{key}={fields[key]}"
            )

    if "Name" in fields:
        parts.append(
            f'Name="{fields["Name"]}"'
        )

    parts.extend(
        [
            f"len={packet['length']}",
            f"corr={packet['score']:.3f}",
            f"CFO={packet['cfo_hz']/1000.0:+.1f}kHz",
            "CRC=OK",
        ]
    )

    print(
        " ".join(parts),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Live BLE LE 1M primary-advertising "
            "decoder for K'amal XTRX CS16 stdin."
        )
    )

    default_decoder = (
        Path(__file__).resolve().with_name(
            "kamal-ble-decode-proto-v2.py"
        )
    )

    parser.add_argument(
        "--decoder",
        default=str(default_decoder),
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=8_000_000.0,
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=39,
    )

    parser.add_argument(
        "--sps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=500.0,
    )

    parser.add_argument(
        "--overlap-ms",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--queue-depth",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--power-window-us",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--burst-db",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--merge-us",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--pre-us",
        type=float,
        default=250.0,
    )

    parser.add_argument(
        "--post-us",
        type=float,
        default=3000.0,
    )

    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.72,
    )

    parser.add_argument(
        "--min-dev-khz",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--max-dev-khz",
        type=float,
        default=450.0,
    )

    parser.add_argument(
        "--max-sync-errors",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.channel not in (37, 38, 39):
        raise SystemExit(
            "ERROR: --channel must be 37, 38, or 39"
        )

    if args.rate <= 0:
        raise SystemExit(
            "ERROR: --rate must be positive"
        )

    if args.chunk_ms <= 0:
        raise SystemExit(
            "ERROR: --chunk-ms must be positive"
        )

    if args.overlap_ms < 0:
        raise SystemExit(
            "ERROR: --overlap-ms cannot be negative"
        )

    decoder = load_decoder(
        args.decoder
    )

    decoder.self_test()

    chunk_samples = int(
        round(
            args.chunk_ms
            * 1e-3
            * args.rate
        )
    )

    overlap_samples = int(
        round(
            args.overlap_ms
            * 1e-3
            * args.rate
        )
    )

    chunk_bytes = (
        chunk_samples * 4
    )

    overlap_bytes = (
        overlap_samples * 4
    )

    input_queue = queue.Queue(
        maxsize=args.queue_depth
    )

    reader_errors = []

    thread = threading.Thread(
        target=reader_worker,
        args=(
            sys.stdin.buffer,
            input_queue,
            chunk_bytes,
            reader_errors,
        ),
        daemon=True,
    )

    thread.start()

    print(
        "=== K'AMAL XTRX LIVE BLE SCANNER PROTOTYPE v1 ===",
        file=sys.stderr,
        flush=True,
    )

    print(
        "SELFTEST: PASS",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Channel: CH{args.channel}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Rate: {args.rate/1e6:.6f} MS/s",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Chunk: {args.chunk_ms:.1f} ms "
        f"({chunk_bytes} bytes)",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Overlap: {args.overlap_ms:.1f} ms",
        file=sys.stderr,
        flush=True,
    )

    print(
        "LIVE_START",
        file=sys.stderr,
        flush=True,
    )

    tail = b""

    recent = []

    total_packets = 0
    chunk_number = 0
    max_queue_seen = 0

    while True:
        item = input_queue.get()

        if item is None:
            break

        stream_start_sample, data = item

        chunk_number += 1

        tail_samples = len(tail) // 4

        combined_base_sample = (
            stream_start_sample
            - tail_samples
        )

        combined = tail + data

        raw = np.frombuffer(
            combined,
            dtype="<i2",
        )

        started = time.monotonic()

        packets, region_count, candidates = (
            analyze_chunk(
                decoder=decoder,
                raw=raw,
                base_sample=combined_base_sample,
                rate=args.rate,
                channel=args.channel,
                sps=args.sps,
                power_window_us=args.power_window_us,
                burst_db=args.burst_db,
                merge_us=args.merge_us,
                pre_us=args.pre_us,
                post_us=args.post_us,
                min_corr=args.min_corr,
                min_dev_khz=args.min_dev_khz,
                max_dev_khz=args.max_dev_khz,
                max_sync_errors=args.max_sync_errors,
            )
        )

        decode_seconds = (
            time.monotonic() - started
        )

        emitted = 0

        for packet in packets:
            if packet_is_duplicate(
                packet,
                recent,
            ):
                continue

            print_packet(
                decoder,
                packet,
                args.channel,
            )

            recent.append(
                (
                    packet["time_s"],
                    packet["pdu"],
                )
            )

            if len(recent) > 256:
                recent = recent[-256:]

            emitted += 1
            total_packets += 1

        queue_depth = input_queue.qsize()

        max_queue_seen = max(
            max_queue_seen,
            queue_depth,
        )

        stream_end_s = (
            (
                stream_start_sample
                + len(data) // 4
            )
            / args.rate
        )

        print(
            f"LIVE_CHUNK "
            f"n={chunk_number} "
            f"stream_end={stream_end_s:.3f}s "
            f"regions={region_count} "
            f"candidates={candidates} "
            f"packets={emitted} "
            f"decode={decode_seconds:.3f}s "
            f"queue={queue_depth}",
            file=sys.stderr,
            flush=True,
        )

        if overlap_bytes:
            keep = min(
                overlap_bytes,
                len(combined),
            )

            keep -= keep % 4

            tail = combined[-keep:]
        else:
            tail = b""

    thread.join()

    if reader_errors:
        raise RuntimeError(
            f"stdin reader failed: "
            f"{reader_errors[0]}"
        )

    print(
        "LIVE_END",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Chunks: {chunk_number}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"CRC-valid packets: {total_packets}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Max queue depth: {max_queue_seen}",
        file=sys.stderr,
        flush=True,
    )

    if total_packets:
        print(
            "RESULT: PASS - live CRC-valid BLE packets decoded",
            file=sys.stderr,
            flush=True,
        )
        return 0

    print(
        "RESULT: no CRC-valid BLE packets decoded",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
