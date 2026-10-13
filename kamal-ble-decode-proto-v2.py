#!/usr/bin/env python3
import argparse
import math
import sys

import numpy as np

ADV_AA = 0x8E89BED6
ADV_CRC_INIT = 0x555555

PDU_NAMES = {
    0x0: "ADV_IND",
    0x1: "ADV_DIRECT_IND",
    0x2: "ADV_NONCONN_IND",
    0x3: "SCAN_REQ",
    0x4: "SCAN_RSP",
    0x5: "CONNECT_IND",
    0x6: "ADV_SCAN_IND",
    0x7: "ADV_EXT_IND",
}


def bytes_to_air_bits(data):
    return np.array(
        [(byte >> bit) & 1 for byte in data for bit in range(8)],
        dtype=np.uint8,
    )


def air_bits_to_bytes(bits):
    count = len(bits) // 8
    return bytes(
        sum(int(bits[8 * n + bit]) << bit for bit in range(8))
        for n in range(count)
    )


def advertising_sync_bits():
    aa = np.array([(ADV_AA >> bit) & 1 for bit in range(32)], dtype=np.uint8)
    preamble = np.array(
        [aa[0] if (bit % 2) == 0 else 1 - aa[0] for bit in range(8)],
        dtype=np.uint8,
    )
    return np.concatenate((preamble, aa))


def whiten_bytes(data, channel):
    """BLE whitening/de-whitening in on-air bit order."""
    out = bytearray(data)
    lfsr = int(channel) | 0x40

    for n in range(len(out)):
        mask_byte = 0
        mask = 1

        for _ in range(8):
            if lfsr & 0x01:
                lfsr ^= 0x88
                mask_byte |= mask

            lfsr >>= 1
            mask <<= 1

        out[n] ^= mask_byte

    return bytes(out)


def advertising_crc(data):
    """BLE advertising CRC-24, CRCInit 0x555555, returned in air-byte form."""
    # In this wire-order representation, CRCInit 0x555555 is 0xAA,0xAA,0xAA.
    reg = [0xAA, 0xAA, 0xAA]

    for value in data:
        d = value

        for _ in range(8):
            feedback = reg[0] & 0x01

            reg[0] >>= 1
            if reg[1] & 0x01:
                reg[0] |= 0x80

            reg[1] >>= 1
            if reg[2] & 0x01:
                reg[1] |= 0x80

            reg[2] >>= 1

            if feedback != (d & 0x01):
                reg[2] ^= 0xDA
                reg[1] ^= 0x60

            d >>= 1

    return bytes(reg)


def self_test():
    # Bluetooth Core sample-data ADV_NONCONN_IND example on channel 38.
    pdu = bytes.fromhex("42 09 a6 a5 a4 a3 a2 c1 01 02 03")
    expected_crc = bytes.fromhex("ad b4 eb")
    expected_whitened = bytes.fromhex(
        "94 cc e2 85 fd 7d 43 4e 1a a7 ac ef cf a5"
    )

    crc = advertising_crc(pdu)
    if crc != expected_crc:
        raise RuntimeError(
            f"CRC self-test failed: {crc.hex()} != {expected_crc.hex()}"
        )

    encoded = whiten_bytes(pdu + crc, 38)
    if encoded != expected_whitened:
        raise RuntimeError(
            "whitening self-test failed: "
            f"{encoded.hex()} != {expected_whitened.hex()}"
        )

    if whiten_bytes(encoded, 38) != pdu + crc:
        raise RuntimeError("de-whitening self-test failed")

    # First 64 whitening bits for channel 39 from the Core sample-data table,
    # represented here as bytes whose bit 0 is transmitted first.
    expected_ch39_mask = bytes.fromhex("1f 37 4a 5f 85 f6 9c 9a")
    actual_ch39_mask = whiten_bytes(bytes(8), 39)

    if actual_ch39_mask != expected_ch39_mask:
        raise RuntimeError(
            "channel-39 whitening self-test failed: "
            f"{actual_ch39_mask.hex()} != {expected_ch39_mask.hex()}"
        )


def moving_corr(values, template):
    n = len(template)
    if len(values) < n:
        return np.empty(0, dtype=np.float64)

    t = template.astype(np.float64)
    t -= t.mean()
    t_energy = float(np.dot(t, t))

    y = values.astype(np.float64, copy=False)
    numerator = np.convolve(y, t[::-1], mode="valid")

    cs = np.concatenate(([0.0], np.cumsum(y, dtype=np.float64)))
    cs2 = np.concatenate(([0.0], np.cumsum(y * y, dtype=np.float64)))

    sy = cs[n:] - cs[:-n]
    sy2 = cs2[n:] - cs2[:-n]
    variance_term = sy2 - (sy * sy) / n

    denom = np.sqrt(np.maximum(variance_term * t_energy, 1e-30))
    return numerator / denom


def local_peaks(score, threshold):
    if len(score) == 0:
        return np.empty(0, dtype=np.int64)

    mask = score >= threshold
    left = np.empty_like(score)
    right = np.empty_like(score)

    left[0] = -np.inf
    left[1:] = score[:-1]
    right[-1] = -np.inf
    right[:-1] = score[1:]

    mask &= score >= left
    mask &= score >= right

    return np.flatnonzero(mask)


def valid_primary_header(pdu_type, length):
    if pdu_type in (0x0, 0x2, 0x4, 0x6):
        return 6 <= length <= 37
    if pdu_type in (0x1, 0x3):
        return length == 12
    if pdu_type == 0x5:
        return length == 34
    if pdu_type == 0x7:
        return 1 <= length <= 255
    return False


def format_address(raw):
    if len(raw) != 6:
        return None
    return ":".join(f"{value:02X}" for value in raw[::-1])


def parse_ad_name(data):
    offset = 0
    short_name = None

    while offset < len(data):
        field_len = data[offset]
        if field_len == 0:
            break

        end = offset + 1 + field_len
        if end > len(data) or field_len < 1:
            break

        ad_type = data[offset + 1]
        value = data[offset + 2 : end]

        if ad_type == 0x09:
            return value.decode("utf-8", errors="replace")
        if ad_type == 0x08 and short_name is None:
            short_name = value.decode("utf-8", errors="replace")

        offset = end

    return short_name


def describe_payload(pdu_type, payload):
    fields = {}

    if pdu_type in (0x0, 0x2, 0x4, 0x6) and len(payload) >= 6:
        fields["AdvA"] = format_address(payload[:6])
        name = parse_ad_name(payload[6:])
        if name:
            fields["Name"] = name

    elif pdu_type == 0x1 and len(payload) >= 12:
        fields["AdvA"] = format_address(payload[:6])
        fields["TargetA"] = format_address(payload[6:12])

    elif pdu_type == 0x3 and len(payload) >= 12:
        fields["ScanA"] = format_address(payload[:6])
        fields["AdvA"] = format_address(payload[6:12])

    elif pdu_type == 0x5 and len(payload) >= 12:
        fields["InitA"] = format_address(payload[:6])
        fields["AdvA"] = format_address(payload[6:12])

    return fields


def measure_window_power(raw, total_complex, win_samples, batch_windows=2000):
    total_windows = total_complex // win_samples
    result = np.empty(total_windows, dtype=np.float64)

    for first_window in range(0, total_windows, batch_windows):
        last_window = min(first_window + batch_windows, total_windows)
        count = last_window - first_window

        first_sample = first_window * win_samples
        last_sample = last_window * win_samples

        i = np.asarray(
            raw[2 * first_sample : 2 * last_sample : 2], dtype=np.float32
        ).reshape(count, win_samples)
        q = np.asarray(
            raw[2 * first_sample + 1 : 2 * last_sample : 2], dtype=np.float32
        ).reshape(count, win_samples)

        mi = i.mean(axis=1, dtype=np.float64)
        mq = q.mean(axis=1, dtype=np.float64)

        energy = (i * i + q * q).mean(axis=1, dtype=np.float64)
        result[first_window:last_window] = energy - mi * mi - mq * mq

    return np.maximum(result, 1e-30)


def merge_window_indexes(indexes, max_gap):
    if len(indexes) == 0:
        return []

    runs = []
    start = int(indexes[0])
    previous = start

    for value in indexes[1:]:
        value = int(value)

        if value - previous <= max_gap:
            previous = value
            continue

        runs.append((start, previous))
        start = value
        previous = value

    runs.append((start, previous))
    return runs


def merge_sample_regions(regions):
    if not regions:
        return []

    regions = sorted(regions)
    merged = [list(regions[0])]

    for start, end in regions[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def decode_region(
    raw,
    region_start,
    region_end,
    rate,
    sps,
    channel,
    min_corr,
    min_dev_hz,
    max_dev_hz,
    max_sync_errors,
):
    i = np.asarray(
        raw[2 * region_start : 2 * region_end : 2], dtype=np.float32
    )
    q = np.asarray(
        raw[2 * region_start + 1 : 2 * region_end : 2], dtype=np.float32
    )

    if len(i) < 512:
        return [], []

    x = (i + 1j * q).astype(np.complex64, copy=False)
    x = x.copy()
    x -= np.complex64(np.mean(x, dtype=np.complex128))

    # Small symmetric low-pass smoother. Its group delay is two input samples.
    xf = (
        x[:-4]
        + 4.0 * x[1:-3]
        + 6.0 * x[2:-2]
        + 4.0 * x[3:-1]
        + x[4:]
    ) * (1.0 / 16.0)

    discriminator = np.angle(xf[1:] * np.conjugate(xf[:-1])).astype(
        np.float32, copy=False
    )

    sync_bits = advertising_sync_bits()
    levels = (2 * sync_bits.astype(np.int8) - 1).astype(np.float64)
    centered_levels = levels - levels.mean()
    level_energy = float(np.dot(centered_levels, centered_levels))

    packets = []
    diagnostics = []

    for phase in range(sps):
        usable = ((len(discriminator) - phase) // sps) * sps
        if usable <= 0:
            continue

        symbols = discriminator[
            phase : phase + usable
        ].reshape(-1, sps).mean(axis=1)

        corr = moving_corr(symbols, levels)
        score = np.abs(corr)

        for index in local_peaks(score, min_corr):
            index = int(index)
            sync_segment = symbols[index : index + len(levels)].astype(
                np.float64, copy=False
            )

            y = sync_segment - sync_segment.mean()
            amplitude = float(np.dot(y, centered_levels) / level_energy)
            offset = float(sync_segment.mean() - amplitude * levels.mean())

            deviation_hz = abs(amplitude) * rate / (2.0 * math.pi)
            cfo_hz = offset * rate / (2.0 * math.pi)

            if not (min_dev_hz <= deviation_hz <= max_dev_hz):
                continue

            polarity = 1.0 if amplitude >= 0.0 else -1.0
            decided_sync = (
                ((sync_segment - offset) * polarity) > 0.0
            ).astype(np.uint8)
            sync_errors = int(np.count_nonzero(decided_sync != sync_bits))

            if sync_errors > max_sync_errors:
                continue

            pdu_symbol = index + len(sync_bits)
            remaining = symbols[pdu_symbol:]

            if len(remaining) < 40:
                continue

            hard_bits = (((remaining - offset) * polarity) > 0.0).astype(
                np.uint8
            )
            air_bytes = air_bits_to_bytes(hard_bits)

            if len(air_bytes) < 5:
                continue

            plain = whiten_bytes(air_bytes, channel)
            pdu_type = plain[0] & 0x0F
            length = plain[1]

            header_ok = valid_primary_header(pdu_type, length)

            sync_sample = region_start + 3 + phase + index * sps
            time_s = sync_sample / rate

            diagnostic = {
                "time_s": time_s,
                "score": float(score[index]),
                "sync_errors": sync_errors,
                "deviation_hz": deviation_hz,
                "cfo_hz": cfo_hz,
                "pdu_type": pdu_type,
                "length": length,
                "header_ok": header_ok,
            }

            if not header_ok:
                diagnostics.append(diagnostic)
                continue

            pdu_length = 2 + length
            packet_length = pdu_length + 3

            if len(plain) < packet_length:
                diagnostic["short"] = True
                diagnostics.append(diagnostic)
                continue

            pdu = plain[:pdu_length]
            received_crc = plain[pdu_length:packet_length]
            expected_crc = advertising_crc(pdu)

            diagnostic["crc_rx"] = received_crc.hex()
            diagnostic["crc_calc"] = expected_crc.hex()
            diagnostics.append(diagnostic)

            if received_crc != expected_crc:
                continue

            payload = pdu[2:]
            fields = describe_payload(pdu_type, payload)

            packets.append(
                {
                    "time_s": time_s,
                    "score": float(score[index]),
                    "sync_errors": sync_errors,
                    "deviation_hz": deviation_hz,
                    "cfo_hz": cfo_hz,
                    "phase": phase,
                    "pdu_type": pdu_type,
                    "length": length,
                    "tx_add": (plain[0] >> 6) & 1,
                    "rx_add": (plain[0] >> 7) & 1,
                    "pdu": pdu,
                    "payload": payload,
                    "crc": received_crc,
                    "fields": fields,
                }
            )

    return packets, diagnostics


def dedupe_packets(packets):
    packets = sorted(packets, key=lambda item: (item["time_s"], item["pdu"]))
    result = []

    for packet in packets:
        duplicate = False

        for previous in result[-8:]:
            if (
                packet["pdu"] == previous["pdu"]
                and abs(packet["time_s"] - previous["time_s"]) < 20e-6
            ):
                duplicate = True
                break

        if not duplicate:
            result.append(packet)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CRC-gated BLE LE 1M primary-advertising decoder for XTRX CS16 IQ"
    )
    parser.add_argument("raw", help="interleaved little-endian int16 IQ file")
    parser.add_argument("--rate", type=float, default=8_000_000.0)
    parser.add_argument("--channel", type=int, default=39)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sps", type=int, default=8)
    parser.add_argument("--power-window-us", type=float, default=100.0)
    parser.add_argument("--burst-db", type=float, default=4.0)
    parser.add_argument("--merge-us", type=float, default=300.0)
    parser.add_argument("--pre-us", type=float, default=250.0)
    parser.add_argument("--post-us", type=float, default=3000.0)
    parser.add_argument("--min-corr", type=float, default=0.72)
    parser.add_argument("--min-dev-khz", type=float, default=60.0)
    parser.add_argument("--max-dev-khz", type=float, default=450.0)
    parser.add_argument("--max-sync-errors", type=int, default=5)
    parser.add_argument("--max-regions", type=int, default=1000)
    args = parser.parse_args()

    if args.rate <= 0 or args.seconds <= 0 or args.sps <= 0:
        raise SystemExit("ERROR: rate, seconds, and sps must be positive")
    if args.channel not in (37, 38, 39):
        raise SystemExit("ERROR: this prototype accepts primary channels 37, 38, or 39")
    if not (0.0 < args.min_corr <= 1.0):
        raise SystemExit("ERROR: --min-corr must be in (0, 1]")

    self_test()

    raw = np.memmap(args.raw, dtype="<i2", mode="r")
    if len(raw) % 2:
        raise SystemExit("ERROR: raw IQ file contains an odd int16 count")

    available_complex = len(raw) // 2
    selected_complex = min(
        available_complex, int(round(args.seconds * args.rate))
    )

    win_samples = int(round(args.power_window_us * 1e-6 * args.rate))
    if win_samples < 8:
        raise SystemExit("ERROR: power window is too short")

    print("=== K'AMAL BLE LE 1M DECODER PROTOTYPE v2 ===")
    print("SELFTEST:          PASS")
    print(f"File:              {args.raw}")
    print(f"Channel:           {args.channel}")
    print(f"Sample rate:       {args.rate / 1e6:.6f} MS/s")
    print(f"Selected samples:  {selected_complex}")
    print(f"Selected duration: {selected_complex / args.rate:.6f} s")
    print(f"Power window:      {win_samples} samples")
    print()

    powers = measure_window_power(raw, selected_complex, win_samples)
    median_power = float(np.median(powers))
    threshold_power = median_power * (10.0 ** (args.burst_db / 10.0))
    burst_indexes = np.flatnonzero(powers >= threshold_power)

    print("=== BURST GATE ===")
    print(f"Windows:           {len(powers)}")
    print(f"Median power:      {10.0 * math.log10(median_power):.2f} dB")
    print(f"Threshold:         +{args.burst_db:.2f} dB")
    print(f"Burst windows:     {len(burst_indexes)}")

    if len(burst_indexes) == 0:
        print("RESULT: no RF burst windows crossed the power gate")
        return 1

    merge_windows = max(
        1, int(round(args.merge_us / args.power_window_us))
    )
    runs = merge_window_indexes(burst_indexes, merge_windows)

    pre_samples = int(round(args.pre_us * 1e-6 * args.rate))
    post_samples = int(round(args.post_us * 1e-6 * args.rate))

    regions = []
    for first_window, last_window in runs:
        start = max(0, first_window * win_samples - pre_samples)
        end = min(
            selected_complex,
            (last_window + 1) * win_samples + post_samples,
        )
        regions.append((start, end))

    regions = merge_sample_regions(regions)

    if len(regions) > args.max_regions:
        raise SystemExit(
            f"ERROR: {len(regions)} burst regions exceeds --max-regions; "
            "raise --burst-db before decoding"
        )

    print(f"Burst regions:     {len(regions)}")
    print()

    all_packets = []
    all_diagnostics = []

    for number, (start, end) in enumerate(regions, 1):
        packets, diagnostics = decode_region(
            raw=raw,
            region_start=start,
            region_end=end,
            rate=args.rate,
            sps=args.sps,
            channel=args.channel,
            min_corr=args.min_corr,
            min_dev_hz=args.min_dev_khz * 1000.0,
            max_dev_hz=args.max_dev_khz * 1000.0,
            max_sync_errors=args.max_sync_errors,
        )

        all_packets.extend(packets)
        all_diagnostics.extend(diagnostics)

        if number % 100 == 0:
            print(
                f"Decoded {number}/{len(regions)} regions: "
                f"CRC-valid={len(all_packets)}"
            )

    packets = dedupe_packets(all_packets)

    print("=== DECODER PIPELINE ===")
    print(f"AA/dev candidates: {len(all_diagnostics)}")
    print(
        "Plausible headers: "
        f"{sum(1 for item in all_diagnostics if item['header_ok'])}"
    )
    print(
        "CRC checks:        "
        f"{sum(1 for item in all_diagnostics if 'crc_rx' in item)}"
    )
    print(f"CRC-valid packets: {len(packets)}")
    print()

    if packets:
        print("=== CRC-VALID BLE PACKETS ===")

        for number, packet in enumerate(packets, 1):
            pdu_name = PDU_NAMES.get(
                packet["pdu_type"], f"TYPE_{packet['pdu_type']}"
            )

            print(
                f"[{number}] t={packet['time_s']:.6f}s "
                f"CH{args.channel} {pdu_name} len={packet['length']} "
                f"corr={packet['score']:.4f} "
                f"syncerr={packet['sync_errors']} "
                f"CFO={packet['cfo_hz']/1e3:+.1f}kHz "
                f"dev~{packet['deviation_hz']/1e3:.1f}kHz CRC=OK"
            )

            for key, value in packet["fields"].items():
                print(f"    {key}: {value}")

            print(f"    PDU: {packet['pdu'].hex(' ')}")
            print(f"    CRC: {packet['crc'].hex(' ')}")

        print()
        print("RESULT: PASS - XTRX capture contains CRC-valid BLE packet(s)")
        return 0

    print("=== STRONGEST FAILED CANDIDATES ===")

    ranked = sorted(
        all_diagnostics,
        key=lambda item: item["score"],
        reverse=True,
    )[:20]

    if not ranked:
        print("none")
    else:
        for item in ranked:
            pdu_name = PDU_NAMES.get(
                item["pdu_type"], f"TYPE_{item['pdu_type']}"
            )
            detail = (
                f"t={item['time_s']:.6f}s "
                f"corr={item['score']:.4f} "
                f"syncerr={item['sync_errors']} "
                f"CFO={item['cfo_hz']/1e3:+.1f}kHz "
                f"dev~{item['deviation_hz']/1e3:.1f}kHz "
                f"header={pdu_name}/{item['length']} "
                f"header_ok={item['header_ok']}"
            )

            if "crc_rx" in item:
                detail += (
                    f" crc_rx={item['crc_rx']}"
                    f" crc_calc={item['crc_calc']}"
                )

            print(detail)

    print()
    print("RESULT: no CRC-valid BLE packets decoded")
    return 1


if __name__ == "__main__":
    sys.exit(main())
