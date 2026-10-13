#define _GNU_SOURCE

#include <xtrx_api.h>

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_DEVICE      "/dev/xtrx0"
#define DEFAULT_FREQ_HZ     2426000000.0
#define DEFAULT_RATE_HZ     8000000.0
#define DEFAULT_BW_HZ       5000000.0
#define DEFAULT_DURATION_S  10.0

#define DEFAULT_LNA_GAIN    20
#define DEFAULT_PGA_GAIN    0
#define DEFAULT_TIA_GAIN    9

#define BLOCK_SAMPLES       65536U
#define RX_START_SKIP       8192U

#define FRAME_MAGIC          "KXTRXIQ1"
#define FRAME_VERSION        1U
#define FRAME_HEADER_SIZE    48U
#define FRAME_FLAG_RETUNE    0x0001U

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -o OUT.raw [options]\n"
        "\n"
        "Options:\n"
        "  -o FILE       output raw CS16 IQ file; use - for stdout (required)\n"
        "  -D DEVICE     XTRX device [default: %s]\n"
        "  -t SECONDS    capture duration [default: %.1f]\n"
        "  -f HZ         RX center frequency [default: %.0f]\n"
        "  -c CHANNEL    BLE primary channel: 37, 38, or 39\n"
        "  -H MS         hop 37 -> 38 -> 39 with this dwell time\n"
        "  -F            frame IQ blocks with KXTRXIQ1 metadata\n"
        "  -r HZ         RX sample rate [default: %.0f]\n"
        "  -b HZ         RX bandwidth [default: %.0f]\n"
        "  -g L:P:T      LNA:PGA:TIA gain [default: %d:%d:%d]\n"
        "  -S EPOCH_NS   wait until absolute CLOCK_REALTIME nanoseconds\n",
        prog,
        DEFAULT_DEVICE,
        DEFAULT_DURATION_S,
        DEFAULT_FREQ_HZ,
        DEFAULT_RATE_HZ,
        DEFAULT_BW_HZ,
        DEFAULT_LNA_GAIN,
        DEFAULT_PGA_GAIN,
        DEFAULT_TIA_GAIN);
}

static int parse_gain(const char *s, int *lna, int *pga, int *tia)
{
    return sscanf(s, "%d:%d:%d", lna, pga, tia) == 3 ? 0 : -1;
}

static double ble_primary_frequency_hz(int channel)
{
    switch (channel) {
    case 37:
        return 2402000000.0;
    case 38:
        return 2426000000.0;
    case 39:
        return 2480000000.0;
    default:
        return 0.0;
    }
}

static void put_le16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
}

static void put_le32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void put_le64(uint8_t *p, uint64_t v)
{
    unsigned i;

    for (i = 0; i < 8; i++)
        p[i] = (uint8_t)(v >> (8U * i));
}

/*
 * KXTRXIQ1 frame header, little-endian:
 *
 *   0x00  char[8]   magic = "KXTRXIQ1"
 *   0x08  uint16    version
 *   0x0a  uint16    header_size
 *   0x0c  uint16    BLE channel (37/38/39, or 0xffff)
 *   0x0e  uint16    flags
 *   0x10  uint64    stream-relative sample start
 *   0x18  uint64    XTRX hardware sample timestamp
 *   0x20  uint32    complex sample count
 *   0x24  uint32    reserved
 *   0x28  uint64    center frequency in Hz
 *
 * Payload immediately follows as interleaved little-endian CS16 IQ.
 */
static int write_frame_header(FILE *out,
                              uint16_t channel,
                              uint16_t flags,
                              uint64_t stream_sample_start,
                              uint64_t xtrx_sample_start,
                              uint32_t sample_count,
                              uint64_t center_frequency_hz)
{
    uint8_t h[FRAME_HEADER_SIZE] = {0};

    memcpy(h, FRAME_MAGIC, 8);

    put_le16(h + 8, FRAME_VERSION);
    put_le16(h + 10, FRAME_HEADER_SIZE);
    put_le16(h + 12, channel);
    put_le16(h + 14, flags);

    put_le64(h + 16, stream_sample_start);
    put_le64(h + 24, xtrx_sample_start);

    put_le32(h + 32, sample_count);
    put_le32(h + 36, 0);

    put_le64(h + 40, center_frequency_hz);

    return fwrite(h, 1, sizeof(h), out) == sizeof(h) ? 0 : -1;
}

static double elapsed_seconds(const struct timespec *a,
                              const struct timespec *b)
{
    return (double)(b->tv_sec - a->tv_sec) +
           (double)(b->tv_nsec - a->tv_nsec) / 1000000000.0;
}

static void format_utc(const struct timespec *ts,
                       char *buf,
                       size_t buflen)
{
    struct tm tm;

    gmtime_r(&ts->tv_sec, &tm);

    snprintf(buf, buflen,
             "%04d-%02d-%02dT%02d:%02d:%02d.%09ldZ",
             tm.tm_year + 1900,
             tm.tm_mon + 1,
             tm.tm_mday,
             tm.tm_hour,
             tm.tm_min,
             tm.tm_sec,
             ts->tv_nsec);
}

static void report_xtrx_error(const char *what, int rc)
{
    int e = rc < 0 ? -rc : rc;

    fprintf(stderr, "ERROR: %s: rc=%d", what, rc);

    if (e > 0)
        fprintf(stderr, " (%s)", strerror(e));

    fputc('\n', stderr);
}

int main(int argc, char **argv)
{
    const char *device = DEFAULT_DEVICE;
    const char *out_path = NULL;

    double freq_hz = DEFAULT_FREQ_HZ;
    double rate_hz = DEFAULT_RATE_HZ;
    double bw_hz = DEFAULT_BW_HZ;
    double duration_s = DEFAULT_DURATION_S;

    int ble_channel = -1;
    double hop_dwell_ms = 0.0;
    unsigned hop_index = 0;
    unsigned hop_count = 0;
    uint64_t hop_dwell_samples = 0;
    uint64_t next_hop_at = 0;
    const int hop_channels[] = {37, 38, 39};

    int lna_gain = DEFAULT_LNA_GAIN;
    int pga_gain = DEFAULT_PGA_GAIN;
    int tia_gain = DEFAULT_TIA_GAIN;

    uint64_t start_epoch_ns = 0;

    struct xtrx_dev *dev = NULL;
    FILE *out = NULL;
    int output_is_stdout = 0;
    int framed_output = 0;
    int16_t *iq = NULL;

    double master_hz = 0.0;
    double actual_rate_hz = 0.0;
    double actual_tx_rate_hz = 0.0;
    double actual_freq_hz = 0.0;
    double actual_bw_hz = 0.0;

    double actual_lna = 0.0;
    double actual_pga = 0.0;
    double actual_tia = 0.0;

    uint64_t target_samples;
    uint64_t total_samples = 0;
    uint64_t output_bytes = 0;
    uint64_t frame_count = 0;
    uint16_t next_frame_flags = FRAME_FLAG_RETUNE;
    uint64_t first_sample_ts = 0;
    uint64_t last_sample_ts = 0;
    uint64_t expected_next_ts = 0;
    uint64_t lost_samples = 0;
    uint64_t timestamp_gap_samples = 0;

    unsigned overruns = 0;
    unsigned zero_fill_events = 0;
    unsigned timestamp_discontinuities = 0;

    struct timespec mono_start;
    struct timespec mono_end;
    struct timespec wall_start;
    struct timespec wall_end;

    char start_utc[64];
    char end_utc[64];

    int running = 0;
    int exit_code = 1;
    int opt;
    int rc;

    while ((opt = getopt(argc, argv, "o:D:t:f:c:H:Fr:b:g:S:h")) != -1) {
        switch (opt) {
        case 'o':
            out_path = optarg;
            break;

        case 'D':
            device = optarg;
            break;

        case 't':
            duration_s = strtod(optarg, NULL);
            break;

        case 'f':
            freq_hz = strtod(optarg, NULL);
            break;

        case 'c':
            ble_channel = atoi(optarg);

            if (ble_primary_frequency_hz(ble_channel) == 0.0) {
                fprintf(stderr,
                        "ERROR: BLE primary channel must be 37, 38, or 39\n");
                return 2;
            }

            break;

        case 'H':
            hop_dwell_ms = strtod(optarg, NULL);

            if (hop_dwell_ms <= 0.0) {
                fprintf(stderr,
                        "ERROR: hop dwell must be greater than zero milliseconds\n");
                return 2;
            }

            break;

        case 'F':
            framed_output = 1;
            break;

        case 'r':
            rate_hz = strtod(optarg, NULL);
            break;

        case 'b':
            bw_hz = strtod(optarg, NULL);
            break;

        case 'g':
            if (parse_gain(optarg,
                           &lna_gain,
                           &pga_gain,
                           &tia_gain) != 0) {
                fprintf(stderr, "ERROR: invalid gain string: %s\n", optarg);
                return 2;
            }
            break;

        case 'S': {
            char *end = NULL;

            errno = 0;
            unsigned long long v = strtoull(optarg, &end, 10);

            if (errno != 0 || !end || *end != '\0') {
                fprintf(stderr,
                        "ERROR: invalid -S epoch nanoseconds: %s\n",
                        optarg);
                return 2;
            }

            start_epoch_ns = (uint64_t)v;
            break;
        }

        case 'h':
        default:
            usage(argv[0]);
            return opt == 'h' ? 0 : 2;
        }
    }

    if (!out_path) {
        fprintf(stderr, "ERROR: -o OUT.raw is required\n\n");
        usage(argv[0]);
        return 2;
    }

    if (hop_dwell_ms > 0.0) {
        if (ble_channel >= 0) {
            fprintf(stderr,
                    "ERROR: -c and -H cannot be used together\n");
            return 2;
        }

        hop_index = 0;
        ble_channel = hop_channels[hop_index];
    }

    if (ble_channel >= 0)
        freq_hz = ble_primary_frequency_hz(ble_channel);

    if (duration_s <= 0.0 ||
        freq_hz <= 0.0 ||
        rate_hz <= 0.0 ||
        bw_hz <= 0.0) {
        fprintf(stderr, "ERROR: frequency/rate/bandwidth/duration must be positive\n");
        return 2;
    }

    fprintf(stderr, "=== K'AMAL XTRX NATIVE CAPTURE PROTOTYPE v1 ===\n");
    fprintf(stderr, "Device:       %s\n", device);
    fprintf(stderr, "Route:        physical RxB via SISO + SWAP_AB\n");
    fprintf(stderr, "Antenna path: RX_H / LNAH\n");

    if (ble_channel >= 0)
        fprintf(stderr, "BLE channel:  %d\n", ble_channel);

    if (hop_dwell_ms > 0.0)
        fprintf(stderr,
                "Hop plan:     37 -> 38 -> 39, %.3f ms dwell\n",
                hop_dwell_ms);

    fprintf(stderr, "Requested FC: %.6f MHz\n", freq_hz / 1e6);
    fprintf(stderr, "Requested SR: %.6f MS/s\n", rate_hz / 1e6);
    fprintf(stderr, "Requested BW: %.6f MHz\n", bw_hz / 1e6);
    fprintf(stderr, "Gain:         %d:%d:%d\n",
            lna_gain, pga_gain, tia_gain);
    fprintf(stderr, "Duration:     %.6f s\n", duration_s);
    fprintf(stderr, "Output:       %s\n", out_path);
    fprintf(stderr, "Format:       %s\n",
            framed_output ? "KXTRXIQ1 framed CS16" : "raw CS16");

    rc = xtrx_open(device, XTRX_O_RESET, &dev);
    if (rc != 0) {
        report_xtrx_error("xtrx_open", rc);
        goto cleanup;
    }

    rc = xtrx_set_samplerate(dev,
                             0.0,
                             rate_hz,
                             0.0,
                             0,
                             &master_hz,
                             &actual_rate_hz,
                             &actual_tx_rate_hz);
    if (rc != 0) {
        report_xtrx_error("xtrx_set_samplerate", rc);
        goto cleanup;
    }

    fprintf(stderr,
            "Clock:        %.6f MHz master / %.6f MS/s RX\n",
            master_hz / 1e6,
            actual_rate_hz / 1e6);

    rc = xtrx_tune(dev,
                   XTRX_TUNE_RX_FDD,
                   freq_hz,
                   &actual_freq_hz);
    if (rc != 0) {
        report_xtrx_error("xtrx_tune RX", rc);
        goto cleanup;
    }

    /*
     * At >2.3 GHz use LMS7002M LNAH.
     * Both RFIC channels are configured, but SISO+SWAP_AB below routes
     * physical channel B into the single RX DMA stream.
     */
    rc = xtrx_set_antenna_ex(dev,
                             XTRX_CH_AB,
                             XTRX_RX_H);
    if (rc != 0) {
        report_xtrx_error("xtrx_set_antenna_ex RX_H", rc);
        goto cleanup;
    }

    rc = xtrx_tune_rx_bandwidth(dev,
                                XTRX_CH_AB,
                                bw_hz,
                                &actual_bw_hz);
    if (rc != 0) {
        report_xtrx_error("xtrx_tune_rx_bandwidth", rc);
        goto cleanup;
    }

    rc = xtrx_set_gain(dev,
                       XTRX_CH_AB,
                       XTRX_RX_LNA_GAIN,
                       lna_gain,
                       &actual_lna);
    if (rc != 0) {
        report_xtrx_error("xtrx_set_gain LNA", rc);
        goto cleanup;
    }

    rc = xtrx_set_gain(dev,
                       XTRX_CH_AB,
                       XTRX_RX_PGA_GAIN,
                       pga_gain,
                       &actual_pga);
    if (rc != 0) {
        report_xtrx_error("xtrx_set_gain PGA", rc);
        goto cleanup;
    }

    rc = xtrx_set_gain(dev,
                       XTRX_CH_AB,
                       XTRX_RX_TIA_GAIN,
                       tia_gain,
                       &actual_tia);
    if (rc != 0) {
        report_xtrx_error("xtrx_set_gain TIA", rc);
        goto cleanup;
    }

    fprintf(stderr, "Actual FC:    %.6f MHz\n", actual_freq_hz / 1e6);
    fprintf(stderr, "Actual BW:    %.6f MHz\n", actual_bw_hz / 1e6);
    fprintf(stderr, "Actual gain:  %.1f:%.1f:%.1f\n",
            actual_lna,
            actual_pga,
            actual_tia);

    iq = malloc((size_t)BLOCK_SAMPLES * 2U * sizeof(*iq));
    if (!iq) {
        fprintf(stderr, "ERROR: malloc failed\n");
        goto cleanup;
    }

    if (strcmp(out_path, "-") == 0) {
        out = stdout;
        output_is_stdout = 1;
    } else {
        out = fopen(out_path, "wb");
        if (!out) {
            fprintf(stderr,
                    "ERROR: fopen(%s): %s\n",
                    out_path,
                    strerror(errno));
            goto cleanup;
        }
    }

    /*
     * Large userspace stdio buffer. At 8 MS/s CS16 the stream is
     * approximately 32 MB/s.
     */
    setvbuf(out, NULL, _IOFBF, 8U * 1024U * 1024U);

    /*
     * Match the known-good test_xtrx setup:
     *
     *   - RX only
     *   - CS16 on wire and host
     *   - SISO
     *   - SWAP_AB so physical RxB feeds the SISO stream
     *   - initial RX timestamp skip of 8192 samples
     */
    xtrx_stop(dev, XTRX_TRX);
    xtrx_stop(dev, XTRX_TRX);

    xtrx_run_params_t params;
    xtrx_run_params_init(&params);

    params.dir = XTRX_RX;
    params.nflags = 0;

    params.rx.wfmt = XTRX_WF_16;
    params.rx.hfmt = XTRX_IQ_INT16;
    params.rx.chs = XTRX_CH_AB;
    params.rx.paketsize = 0;
    params.rx.flags =
        XTRX_RSP_SISO_MODE |
        XTRX_RSP_SWAP_AB;

    params.rx_stream_start = RX_START_SKIP;

    if (hop_dwell_ms > 0.0) {
        hop_dwell_samples =
            (uint64_t)llround(
                actual_rate_hz * hop_dwell_ms / 1000.0
            );

        if (hop_dwell_samples == 0) {
            fprintf(stderr,
                    "ERROR: hop dwell rounds to zero samples\n");
            goto cleanup;
        }

        next_hop_at = hop_dwell_samples;

        fprintf(stderr,
                "Hop dwell:    %" PRIu64 " samples\n",
                hop_dwell_samples);
    }

    if (start_epoch_ns != 0) {
        struct timespec target;
        struct timespec now;

        target.tv_sec =
            (time_t)(start_epoch_ns / UINT64_C(1000000000));
        target.tv_nsec =
            (long)(start_epoch_ns % UINT64_C(1000000000));

        clock_gettime(CLOCK_REALTIME, &now);

        uint64_t now_ns =
            (uint64_t)now.tv_sec * UINT64_C(1000000000) +
            (uint64_t)now.tv_nsec;

        fprintf(stderr,
                "ARMED start_epoch_ns=%" PRIu64 "\n",
                start_epoch_ns);

        if (now_ns > start_epoch_ns) {
            fprintf(stderr,
                    "ERROR: requested start time is already in the past"
                    " by %" PRIu64 " ns\n",
                    now_ns - start_epoch_ns);
            goto cleanup;
        }

        rc = clock_nanosleep(CLOCK_REALTIME,
                             TIMER_ABSTIME,
                             &target,
                             NULL);

        if (rc != 0) {
            fprintf(stderr,
                    "ERROR: clock_nanosleep: %s\n",
                    strerror(rc));
            goto cleanup;
        }
    }

    rc = xtrx_run_ex(dev, &params);
    if (rc != 0) {
        report_xtrx_error("xtrx_run_ex", rc);
        goto cleanup;
    }

    running = 1;

    target_samples =
        (uint64_t)llround(actual_rate_hz * duration_s);

    fprintf(stderr,
            "Target:       %" PRIu64 " complex samples\n",
            target_samples);

    clock_gettime(CLOCK_REALTIME, &wall_start);
    clock_gettime(CLOCK_MONOTONIC_RAW, &mono_start);

    format_utc(&wall_start, start_utc, sizeof(start_utc));

    fprintf(stderr, "Capture UTC:  %s\n", start_utc);
    fprintf(stderr, "CAPTURE_START\n");

    while (total_samples < target_samples) {
        if (hop_dwell_samples > 0 &&
            total_samples == next_hop_at) {

            hop_index = (hop_index + 1U) % 3U;
            ble_channel = hop_channels[hop_index];
            freq_hz = ble_primary_frequency_hz(ble_channel);

            rc = xtrx_tune(dev,
                           XTRX_TUNE_RX_FDD,
                           freq_hz,
                           &actual_freq_hz);

            if (rc != 0) {
                report_xtrx_error("xtrx_tune hop", rc);
                goto cleanup;
            }

            hop_count++;
            next_frame_flags |= FRAME_FLAG_RETUNE;

            fprintf(stderr,
                    "HOP sample=%" PRIu64
                    " channel=%d"
                    " requested_fc=%.6fMHz"
                    " actual_fc=%.6fMHz\n",
                    total_samples,
                    ble_channel,
                    freq_hz / 1e6,
                    actual_freq_hz / 1e6);

            next_hop_at += hop_dwell_samples;
        }

        unsigned request = BLOCK_SAMPLES;

        uint64_t remaining = target_samples - total_samples;

        if (remaining < request)
            request = (unsigned)remaining;

        if (hop_dwell_samples > 0 &&
            next_hop_at > total_samples) {

            uint64_t until_hop =
                next_hop_at - total_samples;

            if (until_hop < request)
                request = (unsigned)until_hop;
        }

        void *buffers[1];
        buffers[0] = iq;

        xtrx_recv_ex_info_t ri;
        memset(&ri, 0, sizeof(ri));

        ri.samples = request;
        ri.buffer_count = 1;
        ri.buffers = buffers;

        /*
         * Ask the low-level driver to report a timeout instead of
         * potentially waiting forever if DMA stops.
         */
        ri.flags = RCVEX_TIMOUT;
        ri.timeout = 1000;

        rc = xtrx_recv_sync_ex(dev, &ri);
        if (rc != 0) {
            report_xtrx_error("xtrx_recv_sync_ex", rc);
            goto cleanup;
        }

        if (ri.out_samples == 0) {
            fprintf(stderr,
                    "ERROR: xtrx_recv_sync_ex returned zero samples\n");
            goto cleanup;
        }

        if (total_samples == 0) {
            first_sample_ts = ri.out_first_sample;
        } else if (ri.out_first_sample != expected_next_ts) {
            timestamp_discontinuities++;

            if (ri.out_first_sample > expected_next_ts)
                timestamp_gap_samples +=
                    ri.out_first_sample - expected_next_ts;

            fprintf(stderr,
                    "RX_TIMESTAMP_DISCONTINUITY"
                    " expected=%" PRIu64
                    " actual=%" PRIu64
                    " delta=%" PRId64 "\n",
                    expected_next_ts,
                    ri.out_first_sample,
                    (int64_t)(ri.out_first_sample - expected_next_ts));
        }

        expected_next_ts =
            ri.out_first_sample + ri.out_samples;

        last_sample_ts = expected_next_ts;

        if (ri.out_events & RCVEX_EVENT_OVERFLOW) {
            overruns++;

            if (ri.out_resumed_at > ri.out_overrun_at)
                lost_samples +=
                    ri.out_resumed_at - ri.out_overrun_at;

            fprintf(stderr,
                    "RX_OVERFLOW at=%" PRIu64
                    " resumed=%" PRIu64 "\n",
                    ri.out_overrun_at,
                    ri.out_resumed_at);
        }

        if (ri.out_events & RCVEX_EVENT_FILLED_ZERO)
            zero_fill_events++;

        if (framed_output) {
            uint16_t frame_channel =
                ble_channel >= 0
                    ? (uint16_t)ble_channel
                    : UINT16_MAX;

            uint64_t center_frequency_hz =
                (uint64_t)llround(actual_freq_hz);

            if (write_frame_header(out,
                                   frame_channel,
                                   next_frame_flags,
                                   total_samples,
                                   ri.out_first_sample,
                                   ri.out_samples,
                                   center_frequency_hz) != 0) {
                fprintf(stderr,
                        "ERROR: frame header write failed: %s\n",
                        strerror(errno));
                goto cleanup;
            }

            frame_count++;
            output_bytes += FRAME_HEADER_SIZE;
        }

        size_t written =
            fwrite(iq,
                   2U * sizeof(int16_t),
                   ri.out_samples,
                   out);

        if (written != ri.out_samples) {
            fprintf(stderr,
                    "ERROR: short write: %zu/%u samples: %s\n",
                    written,
                    ri.out_samples,
                    ferror(out) ? strerror(errno) : "unknown error");
            goto cleanup;
        }

        output_bytes +=
            (uint64_t)ri.out_samples
            * 2U
            * sizeof(int16_t);

        next_frame_flags = 0;

        total_samples += ri.out_samples;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &mono_end);
    clock_gettime(CLOCK_REALTIME, &wall_end);

    format_utc(&wall_end, end_utc, sizeof(end_utc));

    xtrx_stop(dev, XTRX_RX);
    xtrx_stop(dev, XTRX_RX);
    running = 0;

    if (fflush(out) != 0) {
        fprintf(stderr,
                "ERROR: fflush: %s\n",
                strerror(errno));
        goto cleanup;
    }

    if (!output_is_stdout) {
        if (fsync(fileno(out)) != 0) {
            fprintf(stderr,
                    "ERROR: fsync: %s\n",
                    strerror(errno));
            goto cleanup;
        }
    }

    {
        double elapsed =
            elapsed_seconds(&mono_start, &mono_end);

        uint64_t bytes =
            total_samples * 2U * sizeof(int16_t);

        fprintf(stderr, "CAPTURE_END\n");
        fprintf(stderr, "\n=== RESULT ===\n");
        fprintf(stderr, "Status:            OK\n");
        fprintf(stderr, "Start UTC:         %s\n", start_utc);
        fprintf(stderr, "End UTC:           %s\n", end_utc);
        fprintf(stderr, "Samples:           %" PRIu64 "\n",
                total_samples);
        fprintf(stderr, "Bytes:             %" PRIu64 "\n",
                bytes);

        if (framed_output) {
            fprintf(stderr, "Frames:            %" PRIu64 "\n",
                    frame_count);
            fprintf(stderr, "Output bytes:      %" PRIu64 "\n",
                    output_bytes);
        }

        fprintf(stderr, "First sample TS:   %" PRIu64 "\n",
                first_sample_ts);
        fprintf(stderr, "Last sample TS:    %" PRIu64 "\n",
                last_sample_ts);
        fprintf(stderr, "Overruns:          %u\n",
                overruns);
        fprintf(stderr, "Zero-fill events:  %u\n",
                zero_fill_events);
        fprintf(stderr, "Lost samples:      %" PRIu64 "\n",
                lost_samples);
        fprintf(stderr, "TS discontinuities:%u\n",
                timestamp_discontinuities);
        fprintf(stderr, "TS gap samples:    %" PRIu64 "\n",
                timestamp_gap_samples);
        fprintf(stderr, "Hops:              %u\n",
                hop_count);
        fprintf(stderr, "Wall duration:     %.6f s\n",
                elapsed);
        fprintf(stderr, "Effective rate:    %.6f MS/s\n",
                elapsed > 0.0
                    ? ((double)total_samples / elapsed) / 1e6
                    : 0.0);
        fprintf(stderr, "Disk payload rate: %.3f MB/s\n",
                elapsed > 0.0
                    ? ((double)bytes / elapsed) / 1e6
                    : 0.0);
    }

    exit_code = 0;

cleanup:
    if (running && dev) {
        xtrx_stop(dev, XTRX_RX);
        xtrx_stop(dev, XTRX_RX);
    }

    if (out && !output_is_stdout)
        fclose(out);

    free(iq);

    if (dev)
        xtrx_close(dev);

    return exit_code;
}
