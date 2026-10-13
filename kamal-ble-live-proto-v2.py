#!/usr/bin/env python3

import argparse
import importlib.util
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np


FRAME = struct.Struct("<8sHHHHQQIIQ")

FRAME_MAGIC = b"KXTRXIQ1"
FRAME_VERSION = 1
FRAME_HEADER_SIZE = 48
FRAME_FLAG_RETUNE = 0x0001

EXPECTED_FREQ = {
    37: 2402000000,
    38: 2426000000,
    39: 2480000000,
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def read_exact(stream, wanted):
    buf = bytearray(wanted)
    view = memoryview(buf)
    offset = 0

    while offset < wanted:
        count = stream.readinto(
            view[offset:]
        )

        if not count:
            break

        offset += count

    return bytes(view[:offset])


def frame_reader(
    stream,
    output_queue,
    errors,
    stats,
):
    expected_stream_start = 0
    expected_hw_start = None

    try:
        while True:
            header = read_exact(
                stream,
                FRAME_HEADER_SIZE,
            )

            if not header:
                break

            if len(header) != FRAME_HEADER_SIZE:
                raise RuntimeError(
                    "truncated KXTRXIQ1 header: "
                    f"{len(header)}/{FRAME_HEADER_SIZE}"
                )

            (
                magic,
                version,
                header_size,
                channel,
                flags,
                stream_start,
                hw_start,
                sample_count,
                reserved,
                center_hz,
            ) = FRAME.unpack(header)

            stats["frames"] += 1

            if magic != FRAME_MAGIC:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    f"bad magic {magic!r}"
                )

            if version != FRAME_VERSION:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    f"unsupported version {version}"
                )

            if header_size != FRAME_HEADER_SIZE:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    f"header size {header_size}"
                )

            if reserved != 0:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    f"reserved field is {reserved}"
                )

            if channel not in EXPECTED_FREQ:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    f"invalid BLE channel {channel}"
                )

            if center_hz != EXPECTED_FREQ[channel]:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    f"CH{channel} center frequency "
                    f"{center_hz}"
                )

            unknown_flags = (
                flags & ~FRAME_FLAG_RETUNE
            )

            if unknown_flags:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    f"unknown flags 0x{unknown_flags:04x}"
                )

            if sample_count == 0:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    "zero sample count"
                )

            if (
                stream_start
                != expected_stream_start
            ):
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    "stream discontinuity "
                    f"expected={expected_stream_start} "
                    f"actual={stream_start}"
                )

            if expected_hw_start is None:
                expected_hw_start = hw_start

            if hw_start != expected_hw_start:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    "XTRX timestamp discontinuity "
                    f"expected={expected_hw_start} "
                    f"actual={hw_start}"
                )

            payload_bytes = sample_count * 4

            payload = read_exact(
                stream,
                payload_bytes,
            )

            if len(payload) != payload_bytes:
                raise RuntimeError(
                    f"frame {stats['frames']}: "
                    "truncated payload "
                    f"{len(payload)}/{payload_bytes}"
                )

            item = {
                "channel": channel,
                "flags": flags,
                "stream_start": stream_start,
                "hw_start": hw_start,
                "sample_count": sample_count,
                "center_hz": center_hz,
                "payload": payload,
            }

            output_queue.put(item)

            stats["samples"] += sample_count

            if flags & FRAME_FLAG_RETUNE:
                stats["retune_frames"] += 1

            stats["max_queue"] = max(
                stats["max_queue"],
                output_queue.qsize(),
            )

            expected_stream_start = (
                stream_start + sample_count
            )

            expected_hw_start = (
                hw_start + sample_count
            )

    except BaseException as exc:
        errors.append(exc)

    finally:
        stats["final_stream_sample"] = (
            expected_stream_start
        )

        stats["final_hw_sample"] = (
            expected_hw_start
        )

        output_queue.put(None)


def decode_dwell(
    live_core,
    decoder,
    data,
    channel,
    dwell_start_sample,
    rate,
    settle_samples,
    args,
):
    total_samples = len(data) // 4

    if total_samples <= settle_samples:
        return [], 0, 0, 0

    usable = memoryview(data)[
        settle_samples * 4:
    ]

    raw = np.frombuffer(
        usable,
        dtype="<i2",
    )

    packets, regions, candidates = (
        live_core.analyze_chunk(
            decoder=decoder,
            raw=raw,
            base_sample=(
                dwell_start_sample
                + settle_samples
            ),
            rate=rate,
            channel=channel,
            sps=args.sps,
            power_window_us=(
                args.power_window_us
            ),
            burst_db=args.burst_db,
            merge_us=args.merge_us,
            pre_us=args.pre_us,
            post_us=args.post_us,
            min_corr=args.min_corr,
            min_dev_khz=args.min_dev_khz,
            max_dev_khz=args.max_dev_khz,
            max_sync_errors=(
                args.max_sync_errors
            ),
        )
    )

    return (
        packets,
        regions,
        candidates,
        total_samples - settle_samples,
    )


def main():
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Live BLE LE 1M primary-channel "
            "hopping decoder for KXTRXIQ1 stdin."
        )
    )

    parser.add_argument(
        "--live-core",
        default=str(
            here
            / "kamal-ble-live-proto-v1.py"
        ),
    )

    parser.add_argument(
        "--decoder",
        default=str(
            here
            / "kamal-ble-decode-proto-v2.py"
        ),
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=8_000_000.0,
    )

    parser.add_argument(
        "--settle-ms",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--queue-depth",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--require-all-primary",
        action="store_true",
    )

    parser.add_argument(
        "--sps",
        type=int,
        default=8,
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

    if args.rate <= 0:
        raise SystemExit(
            "ERROR: --rate must be positive"
        )

    if args.settle_ms < 0:
        raise SystemExit(
            "ERROR: --settle-ms cannot be negative"
        )

    if args.queue_depth < 1:
        raise SystemExit(
            "ERROR: --queue-depth must be >= 1"
        )

    live_core = load_module(
        args.live_core,
        "kamal_ble_live_proto_v1",
    )

    decoder = load_module(
        args.decoder,
        "kamal_ble_decode_proto_v2",
    )

    decoder.self_test()

    settle_samples = int(
        round(
            args.settle_ms
            * 1e-3
            * args.rate
        )
    )

    input_queue = queue.Queue(
        maxsize=args.queue_depth
    )

    reader_errors = []

    reader_stats = {
        "frames": 0,
        "samples": 0,
        "retune_frames": 0,
        "max_queue": 0,
        "final_stream_sample": 0,
        "final_hw_sample": None,
    }

    thread = threading.Thread(
        target=frame_reader,
        args=(
            sys.stdin.buffer,
            input_queue,
            reader_errors,
            reader_stats,
        ),
        daemon=True,
    )

    thread.start()

    print(
        "=== K'AMAL XTRX LIVE BLE HOPPING "
        "SCANNER PROTOTYPE v2 ===",
        file=sys.stderr,
        flush=True,
    )

    print(
        "SELFTEST: PASS",
        file=sys.stderr,
        flush=True,
    )

    print(
        "Input: KXTRXIQ1",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Rate: {args.rate/1e6:.6f} MS/s",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Post-retune guard: "
        f"{args.settle_ms:.3f} ms "
        f"({settle_samples} samples)",
        file=sys.stderr,
        flush=True,
    )

    print(
        "LIVE_START",
        file=sys.stderr,
        flush=True,
    )

    current_channel = None
    current_start = None
    current_hw_start = None

    dwell_parts = []
    dwell_samples = 0
    dwell_frames = 0

    dwell_count = 0
    total_packets = 0

    by_channel = {
        37: 0,
        38: 0,
        39: 0,
    }

    recent = []

    def finish_dwell():
        nonlocal dwell_count
        nonlocal total_packets
        nonlocal dwell_parts
        nonlocal dwell_samples
        nonlocal dwell_frames
        nonlocal recent

        if current_channel is None:
            return

        data = b"".join(
            dwell_parts
        )

        started = time.monotonic()

        (
            packets,
            regions,
            candidates,
            usable_samples,
        ) = decode_dwell(
            live_core=live_core,
            decoder=decoder,
            data=data,
            channel=current_channel,
            dwell_start_sample=current_start,
            rate=args.rate,
            settle_samples=settle_samples,
            args=args,
        )

        elapsed = (
            time.monotonic() - started
        )

        emitted = 0

        for packet in packets:
            if live_core.packet_is_duplicate(
                packet,
                recent,
            ):
                continue

            live_core.print_packet(
                decoder,
                packet,
                current_channel,
            )

            recent.append(
                (
                    packet["time_s"],
                    packet["pdu"],
                )
            )

            if len(recent) > 512:
                recent = recent[-512:]

            emitted += 1
            total_packets += 1

            by_channel[
                current_channel
            ] += 1

        dwell_count += 1

        end_sample = (
            current_start
            + dwell_samples
        )

        print(
            f"LIVE_DWELL "
            f"n={dwell_count} "
            f"CH{current_channel} "
            f"start={current_start} "
            f"end={end_sample} "
            f"samples={dwell_samples} "
            f"usable={usable_samples} "
            f"frames={dwell_frames} "
            f"regions={regions} "
            f"candidates={candidates} "
            f"packets={emitted} "
            f"decode={elapsed:.3f}s "
            f"queue={input_queue.qsize()}",
            file=sys.stderr,
            flush=True,
        )

        dwell_parts = []
        dwell_samples = 0
        dwell_frames = 0

    stream_error = None

    try:
        while True:
            item = input_queue.get()

            if item is None:
                break

            channel = item["channel"]
            flags = item["flags"]

            is_retune = bool(
                flags & FRAME_FLAG_RETUNE
            )

            if current_channel is None:
                if not is_retune:
                    raise RuntimeError(
                        "first IQ frame does not "
                        "carry RETUNE/start flag"
                    )

                current_channel = channel
                current_start = (
                    item["stream_start"]
                )
                current_hw_start = (
                    item["hw_start"]
                )

            elif is_retune:
                finish_dwell()

                if (
                    item["stream_start"]
                    != current_start
                    + dwell_samples
                ):
                    # dwell_samples was reset by
                    # finish_dwell(), so compare
                    # using the frame's own stream
                    # continuity already enforced
                    # by frame_reader.
                    pass

                current_channel = channel
                current_start = (
                    item["stream_start"]
                )
                current_hw_start = (
                    item["hw_start"]
                )

            elif channel != current_channel:
                raise RuntimeError(
                    "BLE channel changed without "
                    "RETUNE flag: "
                    f"CH{current_channel} -> "
                    f"CH{channel}"
                )

            dwell_parts.append(
                item["payload"]
            )

            dwell_samples += (
                item["sample_count"]
            )

            dwell_frames += 1

        finish_dwell()

    except BaseException as exc:
        stream_error = exc

    thread.join()

    if reader_errors:
        stream_error = reader_errors[0]

    print(
        "LIVE_END",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Frames: {reader_stats['frames']}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Stream samples: "
        f"{reader_stats['samples']}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Retune frames: "
        f"{reader_stats['retune_frames']}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Dwells: {dwell_count}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"CRC-valid packets: "
        f"{total_packets}",
        file=sys.stderr,
        flush=True,
    )

    print(
        "Packets by channel: "
        f"CH37={by_channel[37]} "
        f"CH38={by_channel[38]} "
        f"CH39={by_channel[39]}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Max frame queue depth: "
        f"{reader_stats['max_queue']}",
        file=sys.stderr,
        flush=True,
    )

    if stream_error is not None:
        print(
            f"RESULT: FAIL - {stream_error}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if args.require_all_primary:
        missing = [
            channel
            for channel in (37, 38, 39)
            if by_channel[channel] == 0
        ]

        if missing:
            print(
                "RESULT: FAIL - no CRC-valid "
                "packet on channel(s): "
                + ",".join(
                    str(channel)
                    for channel in missing
                ),
                file=sys.stderr,
                flush=True,
            )
            return 1

    if total_packets == 0:
        print(
            "RESULT: FAIL - no CRC-valid "
            "BLE packets decoded",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        "RESULT: PASS - live CRC-valid "
        "BLE hopping scan succeeded",
        file=sys.stderr,
        flush=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
