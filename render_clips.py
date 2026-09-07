"""
Stage 2 of the viral-clip pipeline.

Renders scored segments to clips in a chosen aspect ratio:
  - 16:9 (default) -> landscape, full frame kept
  - 9:16           -> vertical, dynamic speaker-tracking crop (YuNet + snap-on-cut)
  - 1:1            -> square
Captions are word-by-word CapCut style, burnt in via the ffmpeg ASS engine.

No external/paid APIs. ffmpeg + OpenCV only. Usage:
    python render_clips.py [workdir] [--aspect 16:9|9:16|1:1] [--quality preview|final]
"""
import sys
import re
import json
import argparse
import subprocess
import time
from pathlib import Path

MANIFEST_NAME = "rendered_clips.json"

# aspect -> (target_w, target_h, caption_font_size, caption_margin_v)
ASPECTS = {
    "16:9": (1920, 1080, 72, 95),
    "9:16": (1080, 1920, 96, 300),
    "1:1":  (1080, 1080, 82, 120),
}
DEFAULT_ASPECT = "16:9"
FONT = "Arial Black"
WORDS_PER_LINE = 3
WHITE = r"{\c&HFFFFFF&}"
HIGHLIGHT = r"{\c&H00FFFF&}"
DET_FPS = 5
DET_W = 640
EMA_ALPHA = 0.20
JUMP_FRAC = 0.22
MARGIN_FRAC = 0.18
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def sanitize(text, maxlen=70):
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:maxlen].strip("-") or "clip"


def ffprobe_dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def has_h264_nvenc():
    """Return whether this ffmpeg build advertises the NVENC H.264 encoder."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError:
        return False
    encoders = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and re.search(r"\bh264_nvenc\b", encoders) is not None


def video_encoder_args(use_nvenc, quality="final"):
    if quality == "preview":
        if use_nvenc:
            return [
                "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "hq",
                "-rc", "vbr", "-cq", "29", "-b:v", "0",
                "-profile:v", "high", "-pix_fmt", "yuv420p",
            ]
        return [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "29",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
        ]
    if use_nvenc:
        # p4 is a fast, quality-oriented NVENC preset; CQ 20 suits social clips.
        return [
            "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
            "-rc", "vbr", "-cq", "20", "-b:v", "0",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
    ]


def crop_for(W, H, tw, th):
    """Largest centred crop of the WxH source matching the target aspect."""
    tar = tw / float(th)
    src = W / float(H)
    if abs(src - tar) < 1e-3:
        cw, ch = W, H
    elif src > tar:                      # source wider -> crop width
        ch = H
        cw = int(round(H * tar))
    else:                                # source taller -> crop height
        cw = W
        ch = int(round(W / tar))
    cw -= cw % 2
    ch -= ch % 2
    x0 = max(0, (W - cw) // 2)
    y0 = max(0, (H - ch) // 2)
    return cw, ch, x0, y0


def _make_detector(workdir):
    import cv2
    model = Path(workdir).parent / "yunet.onnx"
    if not model.exists():
        try:
            import urllib.request
            urllib.request.urlretrieve(YUNET_URL, str(model))
        except Exception as e:
            print("  YuNet download failed -> Haar:", e, flush=True)
    if model.exists():
        try:
            return "yunet", cv2.FaceDetectorYN.create(str(model), "", (DET_W, 360), 0.6, 0.3, 5000)
        except Exception as e:
            print("  YuNet init failed -> Haar:", e, flush=True)
    return "haar", cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_track(source, start, dur, W, workdir):
    """Low-res detection pass -> [(t_rel, center_x_source_px or None)]."""
    import cv2
    det_path = Path(workdir) / "_det.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source),
         "-t", f"{dur:.3f}", "-vf", f"fps={DET_FPS},scale={DET_W}:-2",
         "-an", str(det_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kind, det = _make_detector(workdir)
    cap = cv2.VideoCapture(str(det_path))
    dw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or DET_W
    dh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360
    sx = W / float(dw)
    if kind == "yunet":
        det.setInputSize((int(dw), int(dh)))
    track = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / float(DET_FPS)
        cx = None
        if kind == "yunet":
            _, faces = det.detect(frame)
            if faces is not None and len(faces):
                best = max(faces, key=lambda f: f[2] * f[3] * float(f[14]))
                cx = (best[0] + best[2] / 2.0) * sx
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = det.detectMultiScale(gray, 1.1, 5, minSize=(36, 36))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx = (x + w / 2.0) * sx
        track.append((t, cx))
        i += 1
    cap.release()
    try:
        det_path.unlink()
    except OSError:
        pass
    return track


def smooth_track(track, W, crop_w):
    half = crop_w / 2.0
    max_off = half - crop_w * MARGIN_FRAC
    jump = W * JUMP_FRAC
    known = [c for _, c in track if c is not None]
    base = sorted(known)[len(known) // 2] if known else W / 2.0
    filled = []
    last = base
    for t, c in track:
        if c is None:
            c = last
        last = c
        filled.append((t, c))
    if not filled:
        return [(0.0, min(max(base, half), W - half))]
    out = []
    c = filled[0][1]
    prev_face = filled[0][1]
    for t, face in filled:
        if abs(face - prev_face) > jump:
            c = face
        else:
            c += EMA_ALPHA * (face - c)
        if face - c > max_off:
            c = face - max_off
        elif c - face > max_off:
            c = face + max_off
        prev_face = face
        out.append((t, min(max(c, half), W - half)))
    return out


def build_sendcmd(track, crop_w, path):
    lines = []
    last = None
    for t, cx in track:
        x = int(round(cx - crop_w / 2.0))
        if last is None or abs(x - last) >= 2:
            lines.append(f"{t:.2f} crop x {x};")
            last = x
    if not lines:
        lines = ["0.0 crop x 0;"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(words, clip_start, clip_end, path, tw, th, font_size, margin_v):
    sub = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {tw}\nPlayResY: {th}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{FONT},{font_size},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H64000000,-1,0,0,0,100,100,0,0,1,5,2,2,80,80,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    groups = []
    current_group = []
    for word in sub:
        if word.get("group_break") and current_group:
            groups.append(current_group)
            current_group = []
        current_group.append(word)
        if len(current_group) >= WORDS_PER_LINE:
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)
    events = []
    for g in groups:
        for i, w in enumerate(g):
            st = max(w["start"], clip_start) - clip_start
            en = (g[i + 1]["start"] - clip_start) if i + 1 < len(g) \
                else (w["end"] - clip_start)
            if en <= st:
                en = st + 0.12
            parts = []
            for j, ww in enumerate(g):
                token = ass_escape(ww["word"].strip())
                parts.append((HIGHLIGHT + token + WHITE) if j == i else (WHITE + token))
            events.append(
                f"Dialogue: 0,{ass_time(st)},{ass_time(en)},Cap,,0,0,0,,{' '.join(parts)}")
    Path(path).write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def remap_timeline_words(words, periods):
    """Duplicate and shift source words into the cold-open timeline."""
    mapped = []
    for source_start, source_end, timeline_start in periods:
        first_in_period = True
        for word in words:
            word_start = float(word["start"])
            word_end = float(word["end"])
            if word_end <= source_start or word_start >= source_end:
                continue
            mapped.append({
                "word": word["word"],
                "start": timeline_start + max(word_start, source_start) - source_start,
                "end": timeline_start + min(word_end, source_end) - source_start,
                "group_break": first_in_period,
            })
            first_in_period = False
    return mapped


def validated_hook(seg, clip_start, clip_end):
    """Return a safe in-clip hook range, or None when the stored hook is unusable."""
    if not bool(seg.get("hook_enabled")):
        return None
    try:
        hook_start = float(seg.get("hook_start"))
        hook_end = float(seg.get("hook_end"))
    except (TypeError, ValueError):
        return None
    hook_duration = hook_end - hook_start
    if not (clip_start <= hook_start < hook_end <= clip_end):
        return None
    if not 1.25 <= hook_duration <= 3.5:
        return None
    if hook_start - clip_start < 1.0:
        return None
    return hook_start, hook_end


def remove_file(path):
    """Remove a render artifact if it exists, including failed ffmpeg output."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def write_manifest(out_dir, paths):
    """Record only MP4s completed successfully by ffmpeg in this render run."""
    manifest = out_dir / MANIFEST_NAME
    temp_manifest = out_dir / (MANIFEST_NAME + ".tmp")
    temp_manifest.write_text(
        json.dumps([Path(path).name for path in paths], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_manifest.replace(manifest)


def print_render_timing(dur, preparation, ffmpeg_render, post_processing,
                        total, encoder, preset):
    print("--- CLIP TIMING ---", flush=True)
    print(f"Preparation / filter setup: {preparation:.2f}s", flush=True)
    print(f"FFmpeg render: {ffmpeg_render:.2f}s", flush=True)
    print(f"Post-processing / file check: {post_processing:.2f}s", flush=True)
    print(f"Total: {total:.2f}s", flush=True)
    print(f"Clip duration: {dur:.2f}s", flush=True)
    print(f"Used encoder: {encoder}", flush=True)
    print(f"Preset: {preset}", flush=True)
    if dur > 0:
        factor = ffmpeg_render / dur
        if factor >= 1:
            speed = f"{factor:.2f}x slower than realtime"
        elif factor > 0:
            speed = f"{1.0 / factor:.2f}x faster than realtime"
        else:
            speed = "instantaneous"
        print(f"Speed factor: {speed}", flush=True)


def render(seg, words, source, W, H, workdir, out_dir, aspect, encoder_state,
           quality="final"):
    total_started = time.perf_counter()
    final_tw, final_th, font_size, margin_v = ASPECTS[aspect]
    if quality == "preview":
        tw, th = final_tw // 3, final_th // 3
    else:
        tw, th = final_tw, final_th

    start = float(seg["start"])
    end = float(seg["end"])
    main_dur = end - start
    hook_range = validated_hook(seg, start, end)
    hook_dur = (hook_range[1] - hook_range[0]) if hook_range else 0.0
    dur = main_dur + hook_dur

    if hook_range:
        caption_words = remap_timeline_words(
            words,
            [
                (hook_range[0], hook_range[1], 0.0),
                (start, end, hook_dur),
            ],
        )
        caption_start, caption_end = 0.0, dur
    else:
        caption_words = words
        caption_start, caption_end = start, end

    build_ass(
        caption_words,
        caption_start,
        caption_end,
        out_dir / "_caption.ass",
        final_tw,
        final_th,
        font_size,
        margin_v
    )

    # --------------------------------------------------
    # 9:16 FULL VIDEO FIT
    # Komplettes Original bleibt sichtbar.
    # Hintergrund wird vergrößert + weichgezeichnet.
    # --------------------------------------------------

    if aspect == "9:16" and quality == "preview":

        # Blur at half the preview resolution, then upscale it behind the fitted
        # source. This handles only 1/36 as many pixels as the final background.
        vf = (
            "split=2[bg][fg];"
            "[bg]"
            "scale=180:320:force_original_aspect_ratio=increase,"
            "crop=180:320,"
            "boxblur=10:6,"
            "scale=360:640"
            "[background];"
            "[fg]"
            "scale=360:640:force_original_aspect_ratio=decrease"
            "[foreground];"
            "[background][foreground]"
            "overlay=(W-w)/2:(H-h)/2,"
            "subtitles=_caption.ass"
        )

        mode = "vertical-full-video-preview"

    elif aspect == "9:16":

        vf = (
            "split=2[bg][fg];"
            "[bg]"
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "boxblur=30:20"
            "[background];"
            "[fg]"
            "scale=1080:1920:force_original_aspect_ratio=decrease"
            "[foreground];"
            "[background][foreground]"
            "overlay=(W-w)/2:(H-h)/2,"
            "subtitles=_caption.ass"
        )

        mode = "vertical-full-video"

    # --------------------------------------------------
    # 16:9 / 1:1
    # Bisheriges Verhalten
    # --------------------------------------------------

    else:

        cw, ch, x0, y0 = crop_for(
            W,
            H,
            tw,
            th
        )

        needs_track = cw < W * 0.95

        if needs_track:
            if hook_range:
                hook_track = detect_track(
                    source, hook_range[0], hook_dur, W, workdir
                )
                main_track = detect_track(source, start, main_dur, W, workdir)
                raw_track = hook_track + [
                    (time_value + hook_dur, center)
                    for time_value, center in main_track
                ]
            else:
                raw_track = detect_track(source, start, main_dur, W, workdir)
            track = smooth_track(
                raw_track,
                W,
                cw
            )

            build_sendcmd(
                track,
                cw,
                out_dir / "_crop.cmd"
            )

            init_x = max(
                0,
                min(
                    int(
                        round(
                            track[0][1] - cw / 2.0
                        )
                    ),
                    W - cw
                )
            )

            vf = (
                f"sendcmd=f=_crop.cmd,"
                f"crop={cw}:{ch}:{init_x}:{y0},"
                f"scale={tw}:{th},"
                f"subtitles=_caption.ass"
            )

            mode = "speaker-tracked"

        else:

            vf = (
                f"crop={cw}:{ch}:{x0}:{y0},"
                f"scale={tw}:{th},"
                f"subtitles=_caption.ass"
            )

            mode = "full-frame"

    if hook_range:
        mode += "-auto-hook"

    out = out_dir / (
        sanitize(
            seg.get(
                "hook",
                "clip"
            )
        ) + ".mp4"
    )
    temp_out = out.with_name(out.stem + ".part.mp4")
    poster = out.with_suffix(".png")
    temp_poster = poster.with_name(poster.stem + ".part.png")

    if quality == "preview":
        print("PREVIEW MODE", flush=True)
    else:
        print("FINAL MODE", flush=True)
    print(f"OUTPUT RESOLUTION: {tw}x{th}", flush=True)
    print(f"OUTPUT PATH: {out}", flush=True)

    # Never let an old file or a partial ffmpeg output look like this run's result.
    for artifact in (out, temp_out, poster, temp_poster):
        remove_file(artifact)

    if hook_range:
        hook_start, hook_end = hook_range
        filter_complex = (
            f"[0:v]trim=start={hook_start:.3f}:end={hook_end:.3f},"
            "setpts=PTS-STARTPTS[hook_v];"
            f"[0:a]atrim=start={hook_start:.3f}:end={hook_end:.3f},"
            "asetpts=PTS-STARTPTS[hook_a];"
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            "setpts=PTS-STARTPTS[main_v];"
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
            "asetpts=PTS-STARTPTS[main_a];"
            "[hook_v][hook_a][main_v][main_a]"
            "concat=n=2:v=1:a=1[timeline_v][timeline_a];"
            f"[timeline_v]{vf}[video_out];"
            "[timeline_a]anull[audio_out]"
        )
        cmd_prefix = [
            "ffmpeg", "-y", "-i", str(source),
            "-filter_complex", filter_complex,
            "-map", "[video_out]", "-map", "[audio_out]",
        ]
    else:
        cmd_prefix = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{dur:.3f}",
            "-vf",
            vf,
        ]
    audio_bitrate = "112k" if quality == "preview" else "160k"
    cmd_suffix = [
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        "-tag:v",
        "avc1",
        str(temp_out)
    ]

    preparation_elapsed = time.perf_counter() - total_started
    ffmpeg_elapsed = 0.0
    use_nvenc = encoder_state["nvenc"]
    while True:
        encoder = "h264_nvenc" if use_nvenc else "libx264"
        if quality == "preview":
            preset = "p1" if use_nvenc else "ultrafast"
        else:
            preset = "p4" if use_nvenc else "veryfast"
        print(
            f"QUALITY: {quality} | ENCODER: {encoder} | PRESET: {preset} | "
            f"GPU-ENCODING: {'AKTIV' if use_nvenc else 'NEIN'}",
            flush=True,
        )
        cmd = cmd_prefix + video_encoder_args(use_nvenc, quality) + cmd_suffix
        print(">", " ".join(cmd), flush=True)
        ffmpeg_started = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            errors="replace",
        )
        ffmpeg_elapsed += time.perf_counter() - ffmpeg_started
        if result.returncode == 0:
            break

        detail = (result.stderr or result.stdout or "ffmpeg returned no error text").strip()
        remove_file(temp_out)
        if use_nvenc:
            print(
                "WARNING: NVENC rendering failed; falling back to libx264.\n"
                f"NVENC ffmpeg error:\n{detail}",
                file=sys.stderr,
                flush=True,
            )
            encoder_state["nvenc"] = False
            use_nvenc = False
            continue
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}:\n{detail}"
        )
    post_started = time.perf_counter()
    if not temp_out.is_file() or temp_out.stat().st_size <= 0:
        remove_file(temp_out)
        raise RuntimeError(
            "ffmpeg returned success, but the MP4 output is missing or empty"
        )
    temp_out.replace(out)

    if quality == "preview":
        print(f"SAVED PREVIEW {out} ({aspect} {tw}x{th}, {mode})", flush=True)
        post_elapsed = time.perf_counter() - post_started
        total_elapsed = time.perf_counter() - total_started
        print_render_timing(
            dur, preparation_elapsed, ffmpeg_elapsed, post_elapsed,
            total_elapsed, encoder, preset,
        )
        return out

    # A thumbnail is optional. Its failure must not invalidate a good MP4.
    poster_saved = False
    try:
        poster_result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{dur * 0.4:.2f}", "-i", str(out),
                "-frames:v", "1", "-q:v", "2", str(temp_poster)
            ],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if (poster_result.returncode == 0 and temp_poster.is_file()
                and temp_poster.stat().st_size > 0):
            temp_poster.replace(poster)
            poster_saved = True
        else:
            detail = (poster_result.stderr or poster_result.stdout or
                      "poster ffmpeg returned no error text").strip()
            print(f"WARNING poster generation failed for {out.name}:\n{detail}", flush=True)
    except Exception as e:
        print(f"WARNING poster generation failed for {out.name}:\n{e}", flush=True)
    finally:
        remove_file(temp_poster)

    poster_status = f"+ poster {poster.name}" if poster_saved else "+ no poster"
    print(f"SAVED {out} ({aspect} {tw}x{th}, {mode}) {poster_status}", flush=True)

    post_elapsed = time.perf_counter() - post_started
    total_elapsed = time.perf_counter() - total_started
    print_render_timing(
        dur, preparation_elapsed, ffmpeg_elapsed, post_elapsed,
        total_elapsed, encoder, preset,
    )

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--aspect", default=DEFAULT_ASPECT, choices=list(ASPECTS),
                    help="output aspect ratio (default 16:9)")
    ap.add_argument("--quality", default="final", choices=["preview", "final"],
                    help="fast preview or full-quality final render")
    ap.add_argument("--segments-file", default=None,
                    help="optional segments JSON; defaults to <workdir>/segments.json")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    out_dir = workdir / "clips"
    if args.quality == "preview":
        out_dir = out_dir / "previews" / args.aspect.replace(":", "x")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.quality == "final":
        write_manifest(out_dir, [])

    transcript = json.loads((workdir / "transcript.json").read_text(encoding="utf-8"))
    segments_path = Path(args.segments_file).resolve() if args.segments_file \
        else workdir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    words = transcript["words"]

    source = Path(transcript["video"])
    if not source.exists():
        cands = sorted(workdir.glob("source.*"))
        source = cands[0] if cands else source
    source = source.resolve()

    W, H = ffprobe_dims(source)
    print(
        f"Source {source} {W}x{H}; {len(segments)} clips -> "
        f"{args.aspect} ({args.quality})",
        flush=True,
    )

    nvenc_available = has_h264_nvenc()
    print(f"NVENC verfügbar: {'ja' if nvenc_available else 'nein'}", flush=True)
    encoder_state = {"nvenc": nvenc_available}

    saved = []
    failed = []
    for i, seg in enumerate(segments, 1):
        print(f"--- Clip {i}/{len(segments)}: "
              f"{seg.get('hook', '')[:60]!r} (overall {seg.get('overall')}) ---",
              flush=True)
        try:
            rendered = render(
                seg, words, source, W, H, workdir, out_dir, args.aspect,
                encoder_state, args.quality,
            )
            saved.append(str(rendered))
            if args.quality == "final":
                write_manifest(out_dir, saved)
        except Exception as e:
            print(f"ERROR rendering clip {i}:\n{e}", file=sys.stderr, flush=True)
            failed.append(i)

    for temporary in (out_dir / "_caption.ass", out_dir / "_crop.cmd"):
        remove_file(temporary)

    print("RENDERED", len(saved), "clips:", flush=True)
    for s in saved:
        print("  ", s, flush=True)

    if failed:
        raise SystemExit(
            f"Rendering failed for {len(failed)} clip(s): {failed}"
        )


if __name__ == "__main__":
    main()
