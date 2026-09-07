"""
Stage 1 of the viral-clip pipeline.

Downloads a YouTube video at up to Full HD with yt-dlp, then transcribes it
locally with faster-whisper (word-level timestamps). Writes work/transcript.json.

No external/paid APIs. CPU-friendly (int8). Usage:
    python download_and_transcribe.py <youtube_url> --model small
"""
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

TITLE_MARKER = "CHOPIFY_TITLE="


def save_transcription_timing(workdir, elapsed_seconds):
    path = workdir / "timings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(data, dict):
            data = {}
        data["transcription"] = {
            "transcription_seconds": round(float(elapsed_seconds), 3),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"WARNING: Could not save transcription timing: {error}", flush=True)


def configure_unicode_terminal():
    """Use UTF-8 for Windows logs without changing transcript text in memory."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def print_transcript_segment(segment):
    """Never let a console codec failure abort transcription."""
    line = f"[{segment.start:7.2f} -> {segment.end:7.2f}] {segment.text}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_line = line.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
        print(safe_line, flush=True)


configure_unicode_terminal()


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_download(cmd):
    """Run yt-dlp while retaining its real output for retry/error handling."""
    print(">", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(
            result.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
            file=sys.stderr,
            flush=True,
        )
    return result


def is_youtube_retryable_block(result):
    output = f"{result.stdout}\n{result.stderr}".lower()
    markers = (
        "sign in to confirm you're not a bot",
        "sign in to confirm you’re not a bot",
        "sign in to confirm that you're not a bot",
        "login required",
        "authentication required",
        "javascript challenge",
        "js challenge",
        "challenge solving failed",
        "n challenge solving failed",
        "signature solving failed",
        "no supported javascript runtime",
    )
    return any(marker in output for marker in markers)


def extract_youtube_title(output):
    for line in reversed((output or "").splitlines()):
        if line.startswith(TITLE_MARKER):
            encoded_title = line[len(TITLE_MARKER):]
            try:
                title = json.loads(encoded_title)
            except json.JSONDecodeError:
                title = encoded_title
            if isinstance(title, str) and title.strip():
                return title.strip()
    return None


def project_created_at(workdir):
    timestamp = workdir.name.split("_", 1)[0]
    try:
        created = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        created = datetime.now(timezone.utc)
    return created.isoformat().replace("+00:00", "Z")


def save_project_metadata(workdir, url, title):
    metadata = {
        "project_id": workdir.name,
        "title": title or workdir.name,
        "source_url": url,
        "created_at": project_created_at(workdir),
    }
    target = workdir / "metadata.json"
    temporary = workdir / "metadata.json.tmp"
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    print(f"Saved YouTube title: {metadata['title']}", flush=True)


def fetch_youtube_title(url):
    """Fetch the exact yt-dlp title without downloading the video again."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download",
        "--no-playlist",
        "--print", f"{TITLE_MARKER}%(title)j",
        url,
    ]
    print("YouTube title metadata query", flush=True)
    result = run_download(cmd)
    if result.returncode != 0 and is_youtube_retryable_block(result):
        print("Firefox-Cookie-/EJS-Retry für YouTube-Titel", flush=True)
        retry_cmd = cmd[:-1] + [
            "--cookies-from-browser", "firefox",
            "--remote-components", "ejs:github",
            cmd[-1],
        ]
        result = run_download(retry_cmd)

    if result.returncode != 0:
        print(
            "WARNING: Separate YouTube title query failed; "
            "using download metadata fallback.",
            file=sys.stderr,
            flush=True,
        )
        return None
    return extract_youtube_title(result.stdout)


def download(url, workdir):
    out_tmpl = workdir / "source.%(ext)s"
    # Full-HD ceiling for both landscape (1920x1080) and portrait (1080x1920).
    # H.264 is preferred when YouTube offers it because local decoding is much
    # faster than 4K AV1 on many consumer systems. The fallback remains capped.
    format_selector = (
        "bestvideo*[width<=1920][height<=1920]+bestaudio/"
        "best[width<=1920][height<=1920]"
    )
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", format_selector,
        "-S", "res:1080,vcodec:h264,acodec:aac",
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "--no-playlist",
        "--print", f"after_move:{TITLE_MARKER}%(title)j",
        "-o", str(out_tmpl),
        url,
    ]
    print("Normaler Download", flush=True)
    result = run_download(cmd)
    if result.returncode != 0 and is_youtube_retryable_block(result):
        print(
            "Normaler Download fehlgeschlagen: "
            "YouTube Bot-/Login-Sperre oder JavaScript-Challenge.",
            flush=True,
        )
        print("Firefox-Cookie-/EJS-Retry", flush=True)
        retry_cmd = cmd[:-1] + [
            "--cookies-from-browser", "firefox",
            "--remote-components", "ejs:github",
            cmd[-1],
        ]
        result = run_download(retry_cmd)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )

    vids = [p for p in sorted(workdir.glob("source.*"))
            if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v")]
    if not vids:
        raise SystemExit("Download failed: no video file produced.")
    title = fetch_youtube_title(url) or extract_youtube_title(result.stdout)
    save_project_metadata(workdir, url, title)
    return vids[0].resolve()


def transcribe(video, workdir, model_size, device, compute_type):
    from faster_whisper import WhisperModel
    started = time.perf_counter()
    print(f"Loading faster-whisper model={model_size} device={device} "
          f"compute={compute_type} (first run downloads the model)...", flush=True)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(video), word_timestamps=True, vad_filter=True)
    print(f"Detected language: {info.language} "
          f"(p={info.language_probability:.2f}), audio {info.duration:.1f}s", flush=True)

    seg_list, words = [], []
    for seg in segments:
        seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text})
        for w in (seg.words or []):
            words.append({"word": w.word, "start": w.start, "end": w.end})
        print_transcript_segment(seg)

    transcript = {
        "video": str(video),
        "language": info.language,
        "duration": info.duration,
        "model": model_size,
        "segments": seg_list,
        "words": words,
    }
    out = workdir / "transcript.json"
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    save_transcription_timing(workdir, time.perf_counter() - started)
    print(f"Wrote {out}  ({len(words)} words, {len(seg_list)} segments)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--model", default="medium",
                    help="faster-whisper model: tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    video = download(args.url, workdir)
    print("Downloaded:", video, flush=True)
    transcribe(video, workdir, args.model, args.device, args.compute_type)
    print("STAGE 1 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
