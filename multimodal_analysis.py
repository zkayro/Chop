"""Lightweight local audio/video feature extraction for clip discovery.

The analyzer deliberately uses measurable signals rather than claiming emotion
recognition. It samples video at roughly 1 fps, adds 2 fps samples around audio
events, and writes an explainable ``analysis_features.json`` per project.
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from semantic_candidates import update_semantic_features


ANALYSIS_VERSION = 2
MIN_CLIP_SECONDS = 15.0
MAX_CLIP_SECONDS = 60.0
MAX_RAW_CANDIDATES = 50
MAX_SELECTED_CANDIDATES = 20
TWO_STAGE_THRESHOLD_SECONDS = 10 * 60
ANCHOR_CLUSTER_DISTANCE_SECONDS = 10.0
MAX_ANCHOR_CLUSTER_SPAN_SECONDS = 32.0
MAX_VISUAL_REGIONS = 50
VISUAL_BUDGET_RATIO = 0.25
REGION_LEAD_SECONDS = 12.0
REGION_TRAIL_SECONDS = 28.0
MAX_REGION_SECONDS = 55.0


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _analysis_fingerprint(source, transcript_path):
    source_stat = source.stat()
    transcript_stat = transcript_path.stat()
    return {
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "transcript_size": transcript_stat.st_size,
        "transcript_mtime_ns": transcript_stat.st_mtime_ns,
    }


def _cache_matches(data, fingerprint, required_keys):
    return bool(
        isinstance(data, dict)
        and data.get("analysis_version") == ANALYSIS_VERSION
        and data.get("fingerprint") == fingerprint
        and all(key in data for key in required_keys)
    )


def _update_timings(project_dir, values):
    path = project_dir / "timings.json"
    data = _read_json(path) or {}
    current = data.get("analysis")
    current = current if isinstance(current, dict) else {}
    current.update({key: round(float(value), 3) for key, value in values.items()})
    current["completed_at"] = datetime.now(timezone.utc).isoformat()
    data["analysis"] = current
    _atomic_json(path, data)


def _find_source(project_dir, transcript):
    transcript_video = Path(str(transcript.get("video", "")))
    if transcript_video.is_file():
        return transcript_video
    sources = sorted(
        path for path in project_dir.glob("source.*")
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
    )
    return sources[0] if sources else None


def _clean_segments(transcript):
    clean = []
    for segment in transcript.get("segments", []):
        start = float(segment.get("start", 0) or 0)
        end = float(segment.get("end", 0) or 0)
        text = str(segment.get("text", "")).strip()
        if text and end > start:
            clean.append({"start": start, "end": end, "text": text})
    return clean


def _text_interest(text):
    lower = " ".join((text or "").lower().split())
    score = 0.12
    if "?" in text:
        score += 0.18
    if "!" in text:
        score += 0.12
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", lower):
        score += 0.1
    strong = (
        "here's why", "the truth", "the problem", "biggest mistake",
        "you need to", "never", "always", "impossible", "but ",
        "hier ist", "die wahrheit", "das problem", "größte fehler",
        "du musst", "niemals", "immer", "unmöglich", "aber ",
        "secret", "reveal", "surprise", "shocking", "geheimnis",
        "überrasch", "krass", "verrückt", "crazy", "insane",
    )
    score += min(0.4, sum(0.09 for term in strong if term in lower))
    if lower.startswith(("why ", "how ", "what ", "warum ", "wie ", "was ")):
        score += 0.14
    return round(min(1.0, score), 4)


def analyze_audio(video_path, transcript, project_dir):
    """Extract energy, silence and transcript-aligned speech-rate signals."""
    try:
        import numpy as np
    except ImportError as error:
        return [], [], {"available": False, "reason": f"numpy unavailable: {error}"}

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [], [], {"available": False, "reason": "ffmpeg not found on PATH"}

    sample_rate = 16000
    window_seconds = 0.5
    window_samples = int(sample_rate * window_seconds)
    pcm_path = None
    samples = None
    started = time.perf_counter()
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix="multimodal_audio_", suffix=".pcm", dir=project_dir, delete=False
        )
        pcm_path = Path(handle.name)
        handle.close()
        result = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-f", "s16le", str(pcm_path),
            ],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if result.returncode != 0 or not pcm_path.is_file() or pcm_path.stat().st_size == 0:
            detail = (result.stderr or result.stdout or "audio extraction failed").strip()
            return [], [], {"available": False, "reason": detail[-1200:]}

        samples = np.memmap(pcm_path, dtype=np.int16, mode="r")
        window_count = len(samples) // window_samples
        if window_count == 0:
            return [], [], {"available": False, "reason": "audio stream is empty"}

        rms = np.empty(window_count, dtype=np.float32)
        for index in range(window_count):
            chunk = samples[index * window_samples:(index + 1) * window_samples]
            normalized = chunk.astype(np.float32) / 32768.0
            rms[index] = math.sqrt(float(np.mean(normalized * normalized)) + 1e-12)

        low, median, high = np.percentile(rms, [15, 50, 95])
        span = max(float(high - low), 1e-6)
        energy = np.clip((rms - low) / span, 0.0, 1.0)
        delta = np.diff(energy, prepend=energy[0])
        spike_threshold = max(0.72, float(np.percentile(energy, 90)))

        energy_samples = []
        for index in range(0, window_count, 2):
            value = float(np.max(energy[index:index + 2]))
            energy_samples.append({
                "time": round(index * window_seconds, 2),
                "audio_energy": round(value, 4),
            })

        events = []
        for index, value in enumerate(energy):
            if value >= spike_threshold and delta[index] >= 0.16:
                events.append({
                    "time": round(index * window_seconds, 2),
                    "type": "energy_spike",
                    "audio_energy": round(float(value), 4),
                    "sudden_change": round(float(delta[index]), 4),
                })

        silence_mask = energy <= 0.055
        silence_start = None
        for index, silent in enumerate(silence_mask):
            if silent and silence_start is None:
                silence_start = index
            at_end = index == len(silence_mask) - 1
            if silence_start is not None and ((not silent) or at_end):
                stop = index if not silent else index + 1
                duration = (stop - silence_start) * window_seconds
                if duration >= 1.25:
                    events.append({
                        "time": round(silence_start * window_seconds, 2),
                        "end": round(stop * window_seconds, 2),
                        "type": "silence",
                        "duration": round(duration, 2),
                    })
                silence_start = None

        segments = _clean_segments(transcript)
        rates = [
            len(segment["text"].split()) / max(segment["end"] - segment["start"], 0.25)
            for segment in segments
        ]
        median_rate = float(np.median(rates)) if rates else 0.0
        previous_rate = None
        laughter_pattern = re.compile(
            r"(?:\b(?:ha){2,}\b|\b(?:haha|hahaha|laughs?|laughter|lacht|lachen)\b)", re.I
        )
        for segment, speech_rate in zip(segments, rates):
            start_index = max(0, int(segment["start"] / window_seconds))
            end_index = min(window_count, max(start_index + 1, int(math.ceil(segment["end"] / window_seconds))))
            segment_energy = energy[start_index:end_index]
            mean_energy = float(np.mean(segment_energy)) if len(segment_energy) else 0.0
            peak_energy = float(np.max(segment_energy)) if len(segment_energy) else 0.0
            rapid = speech_rate >= max(3.6, median_rate * 1.35)
            rate_change = (
                abs(speech_rate - previous_rate) / max(previous_rate, 0.5)
                if previous_rate is not None else 0.0
            )
            if rapid or rate_change >= 0.45 or peak_energy >= 0.84:
                events.append({
                    "time": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "type": "speech_emphasis",
                    "speech_rate_wps": round(speech_rate, 3),
                    "rapid_speech": bool(rapid),
                    "rate_change": round(rate_change, 3),
                    "mean_energy": round(mean_energy, 4),
                    "peak_energy": round(peak_energy, 4),
                })
            if laughter_pattern.search(segment["text"]) and peak_energy >= 0.45:
                events.append({
                    "time": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "type": "laughter_like",
                    "basis": "transcript marker plus measured audio energy",
                    "peak_energy": round(peak_energy, 4),
                })
            previous_rate = speech_rate

        events.sort(key=lambda item: (item["time"], item["type"]))
        return energy_samples, events, {
            "available": True,
            "sample_rate": sample_rate,
            "energy_window_seconds": window_seconds,
            "median_rms": round(float(median), 7),
            "median_speech_rate_wps": round(median_rate, 3),
            "speaker_change_detection": "not emitted without reliable diarization",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return [], [], {"available": False, "reason": str(error)}
    finally:
        mmap_handle = getattr(samples, "_mmap", None)
        if mmap_handle is not None:
            try:
                mmap_handle.close()
            except OSError:
                pass
        if pcm_path is not None:
            try:
                pcm_path.unlink(missing_ok=True)
            except OSError:
                pass


def _sample_video_frames_ffmpeg(video_path, duration, target_times, np):
    """Decode once in FFmpeg and retain only requested half-second samples."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH")
    width, height, sample_fps = 320, 180, 2.0
    wanted_slots = {max(0, int(round(value * sample_fps))) for value in target_times}
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-hwaccel", "auto", "-i", str(video_path), "-an", "-sn", "-dn",
        "-vf", (
            f"fps={sample_fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=gray"
        ),
        "-t", f"{max(0.1, float(duration) + 0.5):.3f}",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=width * height * 4
    )
    frames = []
    frame_size = width * height
    frame_index = 0
    try:
        while True:
            frame_bytes = process.stdout.read(frame_size)
            if not frame_bytes:
                break
            if len(frame_bytes) != frame_size:
                raise RuntimeError("FFmpeg returned a partial raw video frame")
            if frame_index in wanted_slots:
                gray = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width).copy()
                frames.append((frame_index / sample_fps, gray))
            frame_index += 1
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(stderr or f"FFmpeg exited with code {return_code}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return frames


def _sample_video_regions_ffmpeg(video_path, regions, target_times, np):
    """Read merged regions with input seeking; no region decodes from video start."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH")
    width, height, sample_fps = 320, 180, 2.0
    frames = []
    for region in regions:
        region_start = max(0.0, float(region["start"]))
        region_end = max(region_start, float(region["end"]))
        wanted_slots = {
            max(0, int(round((value - region_start) * sample_fps)))
            for value in target_times
            if region_start <= value <= region_end
        }
        if not wanted_slots:
            continue
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-hwaccel", "auto", "-ss", f"{region_start:.3f}",
            "-i", str(video_path), "-an", "-sn", "-dn",
            "-vf", (
                f"fps={sample_fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=gray"
            ),
            "-t", f"{max(0.1, region_end - region_start + 0.5):.3f}",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=width * height * 4,
        )
        frame_size = width * height
        frame_index = 0
        try:
            while True:
                frame_bytes = process.stdout.read(frame_size)
                if not frame_bytes:
                    break
                if len(frame_bytes) != frame_size:
                    raise RuntimeError("FFmpeg returned a partial raw video frame")
                if frame_index in wanted_slots:
                    gray = np.frombuffer(
                        frame_bytes, dtype=np.uint8
                    ).reshape(height, width).copy()
                    frames.append((region_start + frame_index / sample_fps, gray))
                frame_index += 1
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(stderr or f"FFmpeg exited with code {return_code}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    frames.sort(key=lambda item: item[0])
    return frames


def _sample_video_frames_opencv(cap, target_times, cv2):
    """Fallback that seeks directly instead of decoding every intermediate frame."""
    frames = []
    for sample_time in sorted(target_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(sample_time)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        scale = min(320.0 / max(width, 1), 180.0 / max(height, 1))
        small = cv2.resize(
            frame,
            (max(2, int(width * scale)), max(2, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        frames.append((float(sample_time), cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))
    return frames


def analyze_video(video_path, duration, audio_events, regions=None):
    """Sample lightweight visual-change, motion, zoom and face-count signals."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        return [], {"available": False, "reason": f"OpenCV unavailable: {error}"}

    started = time.perf_counter()
    open_cv_path = str(getattr(cv2, "__file__", "unknown"))
    open_cv_version = str(getattr(cv2, "__version__", "unknown"))
    has_cascade_classifier = hasattr(cv2, "CascadeClassifier")
    print(f"OPEN_CV_PATH: {open_cv_path}", flush=True)
    print(f"OPEN_CV_VERSION: {open_cv_version}", flush=True)
    print(
        "HAS_CASCADE_CLASSIFIER: "
        f"{'true' if has_cascade_classifier else 'false'}",
        flush=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"available": False, "reason": "video could not be opened"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        cap.release()
        return [], {"available": False, "reason": "video FPS unavailable"}
    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps

    analysis_ranges = regions or [{"start": 0.0, "end": duration}]
    target_times = set()
    for region in analysis_ranges:
        target_times.update(
            round(value, 2)
            for value in _frange(float(region["start"]), float(region["end"]), 1.0)
        )
    interesting_audio_times = [
        float(event["time"]) for event in audio_events
        if event.get("type") in {"energy_spike", "speech_emphasis", "laughter_like"}
    ]
    for event_time in interesting_audio_times:
        for offset in (-1.0, -0.5, 0.5, 1.0):
            sample_time = event_time + offset
            if any(
                float(region["start"]) <= sample_time <= float(region["end"])
                for region in analysis_ranges
            ):
                target_times.add(round(sample_time, 2))
    target_frames = sorted({max(0, int(value * fps)) for value in target_times})

    sampling_method = "ffmpeg_region_seek" if regions else "ffmpeg_2fps_stream"
    try:
        if regions:
            sampled_frames = _sample_video_regions_ffmpeg(
                video_path, regions, target_times, np
            )
        else:
            sampled_frames = _sample_video_frames_ffmpeg(
                video_path, duration, target_times, np
            )
        cap.release()
        print(
            f"VIDEO SAMPLER: FFmpeg stream ({len(sampled_frames)} selected frames)",
            flush=True,
        )
    except Exception as sampling_error:
        sampling_method = "opencv_direct_seek"
        print(
            "VIDEO SAMPLER FALLBACK: OpenCV direct seek: "
            f"{type(sampling_error).__name__}: {sampling_error}",
            flush=True,
        )
        sampled_frames = _sample_video_frames_opencv(cap, target_times, cv2)
        cap.release()

    face_detector = None
    face_disabled_reason = None
    if not has_cascade_classifier:
        face_disabled_reason = "CascadeClassifier unavailable"
    else:
        try:
            cascade_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
            if not cascade_root:
                raise RuntimeError("Haar cascade data unavailable")
            face_detector = cv2.CascadeClassifier(
                cascade_root + "haarcascade_frontalface_default.xml"
            )
            if hasattr(face_detector, "empty") and face_detector.empty():
                raise RuntimeError("Haar cascade could not be loaded")
        except Exception as error:
            face_detector = None
            face_disabled_reason = f"{type(error).__name__}: {error}"
    if face_disabled_reason:
        print(f"FACE DETECTION DISABLED: {face_disabled_reason}", flush=True)

    events = []
    feature_errors = {"frame_preprocessing": 0, "scene": 0, "motion": 0, "zoom": 0, "face": 0}
    previous_gray = None
    previous_hist = None
    previous_faces = None
    previous_sample_time = None
    for sample_time, gray in sampled_frames:
        if previous_sample_time is not None and sample_time - previous_sample_time > 2.1:
            previous_gray = None
            previous_hist = None
            previous_faces = None

        hist = None
        try:
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
        except Exception:
            feature_errors["scene"] += 1

        face_count = 0
        face_available = face_detector is not None
        if face_available:
            try:
                faces = face_detector.detectMultiScale(
                    gray, scaleFactor=1.12, minNeighbors=5, minSize=(24, 24)
                )
                face_count = len(faces)
            except Exception as error:
                feature_errors["face"] += 1
                face_detector = None
                face_available = False
                face_disabled_reason = f"{type(error).__name__}: {error}"
                print(f"FACE DETECTION DISABLED: {face_disabled_reason}", flush=True)

        if previous_gray is not None and previous_gray.shape == gray.shape:
            abs_change = 0.0
            try:
                abs_change = float(np.mean(cv2.absdiff(previous_gray, gray)) / 255.0)
            except Exception:
                feature_errors["motion"] += 1
            hist_change = 0.0
            if previous_hist is not None and hist is not None:
                try:
                    hist_change = float(
                        cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    )
                except Exception:
                    feature_errors["scene"] += 1
            motion_score = min(1.0, abs_change * 4.2)
            visual_change = min(1.0, hist_change * 0.72 + abs_change * 2.1)
            scene_change = hist_change >= 0.48 or visual_change >= 0.72
            zoom_scale = 1.0
            if not scene_change:
                try:
                    zoom_scale = _estimate_zoom(cv2, np, previous_gray, gray)
                except Exception:
                    feature_errors["zoom"] += 1
            face_change = (
                face_available
                and previous_faces is not None
                and face_count != previous_faces
            )

            if (
                scene_change or motion_score >= 0.54 or visual_change >= 0.58
                or abs(zoom_scale - 1.0) >= 0.035 or face_change
            ):
                events.append({
                    "time": round(sample_time, 2),
                    "scene_change": bool(scene_change),
                    "motion_score": round(motion_score, 4),
                    "visual_change": round(visual_change, 4),
                    "possible_zoom": bool(abs(zoom_scale - 1.0) >= 0.035),
                    "zoom_scale": round(float(zoom_scale), 4),
                    "face_count": int(face_count),
                    "face_count_change": bool(face_change),
                })
        previous_gray = gray
        previous_hist = hist
        previous_faces = face_count if face_available else None
        previous_sample_time = sample_time

    return events, {
        "available": True,
        "base_sample_fps": 1.0,
        "event_detail_sample_fps": 2.0,
        "requested_frames": len(target_frames),
        "sampled_frames": len(sampled_frames),
        "sampling_method": sampling_method,
        "analyzed_ranges": analysis_ranges,
        "analyzed_seconds": round(
            sum(float(region["end"]) - float(region["start"]) for region in analysis_ranges),
            3,
        ),
        "source_fps": round(fps, 3),
        "open_cv_path": open_cv_path,
        "open_cv_version": open_cv_version,
        "has_cascade_classifier": has_cascade_classifier,
        "face_detection_available": face_detector is not None,
        "face_detection_disabled_reason": face_disabled_reason,
        "feature_errors": feature_errors,
        "face_signal": (
            "face presence/count changes only; no emotion classification"
            if face_detector is not None
            else "disabled"
        ),
        "large_text_detection": "not emitted without a reliable local OCR model",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def _frange(start, stop, step):
    value = start
    while value <= stop:
        yield value
        value += step


def _estimate_zoom(cv2, np, previous_gray, gray):
    points = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=70, qualityLevel=0.02, minDistance=7
    )
    if points is None or len(points) < 12:
        return 1.0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
    if tracked is None or status is None:
        return 1.0
    mask = status.reshape(-1).astype(bool)
    source_points = points.reshape(-1, 2)[mask]
    target_points = tracked.reshape(-1, 2)[mask]
    if len(source_points) < 10:
        return 1.0
    matrix, inliers = cv2.estimateAffinePartial2D(
        source_points, target_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 8:
        return 1.0
    scale = math.sqrt(float(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))
    return scale if 0.85 <= scale <= 1.18 else 1.0


def generate_interest_anchors(segments, energy_samples, audio_events):
    """Find cheap transcript/audio anchors before any expensive video work."""
    anchors = []
    numeric_pattern = re.compile(
        r"(?:[$€£]\s?\d|\b\d+(?:[.,]\d+)?\s?(?:%|percent|prozent|million|millionen|"
        r"billion|milliarden|dollar|euro)\b)", re.I
    )
    novelty_terms = (
        "truth", "actually", "nobody", "most people", "secret", "surpris", "shocking",
        "wahrheit", "eigentlich", "niemand", "die meisten", "geheim", "überrasch", "krass",
    )
    story_terms = (
        "then", "suddenly", "until", "that moment", "realized", "story",
        "dann", "plötzlich", "bis ", "in dem moment", "gemerkt", "geschichte",
    )
    conflict_terms = (
        "but", "wrong", "mistake", "problem", "hate", "fight", "failed", "impossible",
        "aber", "falsch", "fehler", "problem", "hasse", "streit", "gescheitert", "unmöglich",
    )

    def events_near(start, end, radius=1.75):
        return [
            event for event in audio_events
            if start - radius <= float(event.get("time", 0)) <= end + radius
        ]

    for index, segment in enumerate(segments):
        previous_text = segments[index - 1]["text"] if index else ""
        next_text = segments[index + 1]["text"] if index + 1 < len(segments) else ""
        text = segment["text"]
        lower = " ".join(text.lower().split())
        context = " ".join((previous_text, text, next_text))
        duration = max(0.25, segment["end"] - segment["start"])
        nearby = events_near(segment["start"], segment["end"])
        energy_values = [
            float(item.get("audio_energy", 0)) for item in energy_samples
            if segment["start"] - 1 <= float(item.get("time", 0)) <= segment["end"] + 1
        ]
        text_interest = max(_text_interest(text), _text_interest(context) * 0.85)
        numeric_signal = 1.0 if numeric_pattern.search(text) else 0.0
        question_answer_signal = min(
            1.0,
            (0.65 if "?" in text else 0.0)
            + (0.5 if "?" in previous_text and len(text.split()) >= 4 else 0.0)
            + (0.25 if "?" in text and len(next_text.split()) >= 4 else 0.0),
        )
        novelty_signal = min(1.0, sum(term in lower for term in novelty_terms) * 0.34)
        story_signal = min(1.0, sum(term in lower for term in story_terms) * 0.34)
        conflict_signal = min(1.0, sum(re.search(rf"\b{re.escape(term)}", lower) is not None for term in conflict_terms) * 0.3)
        density_signal = min(1.0, len(text.split()) / duration / 3.8)
        energy_signal = max(energy_values, default=0.0)
        event_signal = 0.0
        for event in nearby:
            event_signal = max(event_signal, {
                "energy_spike": 0.82,
                "speech_emphasis": 0.7 + (0.18 if event.get("rapid_speech") else 0.0),
                "laughter_like": 0.9,
                "silence": 0.28,
            }.get(event.get("type"), 0.2))
        audio_interest = min(1.0, event_signal * 0.7 + energy_signal * 0.3)
        score = min(1.0,
            text_interest * 0.28 + audio_interest * 0.2 + novelty_signal * 0.1
            + numeric_signal * 0.1 + question_answer_signal * 0.12
            + story_signal * 0.07 + conflict_signal * 0.08 + density_signal * 0.05
        )
        if score < 0.26:
            continue
        anchors.append({
            "time": round((segment["start"] + segment["end"]) / 2, 3),
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
            "segment_index": index,
            "text": text,
            "interest_score": round(score, 4),
            "modalities": ["text"] + (["audio"] if audio_interest >= 0.3 else []),
            "components": {
                "text_interest": round(text_interest, 4),
                "audio_interest": round(audio_interest, 4),
                "novelty_signal": round(novelty_signal, 4),
                "numeric_signal": round(numeric_signal, 4),
                "question_answer_signal": round(question_answer_signal, 4),
                "story_signal": round(story_signal, 4),
                "conflict_signal": round(conflict_signal, 4),
                "energy_signal": round(energy_signal, 4),
                "density_signal": round(density_signal, 4),
            },
        })

    covered_times = [anchor["time"] for anchor in anchors]
    for event in audio_events:
        if event.get("type") not in {"energy_spike", "speech_emphasis", "laughter_like"}:
            continue
        event_time = float(event.get("time", 0))
        if any(abs(event_time - value) <= 2.0 for value in covered_times):
            continue
        event_score = {
            "energy_spike": 0.48,
            "speech_emphasis": 0.44,
            "laughter_like": 0.58,
        }[event["type"]]
        anchors.append({
            "time": round(event_time, 3), "start": round(event_time, 3),
            "end": round(float(event.get("end", event_time)), 3),
            "segment_index": None, "text": "", "interest_score": event_score,
            "modalities": ["audio"],
            "components": {"audio_interest": event_score},
        })
    # Keep strong anchors across the whole timeline instead of letting an early
    # dense section consume the complete cap on long podcasts.
    bucketed = {}
    for anchor in anchors:
        bucket = int(anchor["time"] // 60.0)
        bucketed.setdefault(bucket, []).append(anchor)
    distributed = []
    for bucket in sorted(bucketed):
        distributed.extend(sorted(
            bucketed[bucket], key=lambda item: item["interest_score"], reverse=True
        )[:4])
    return sorted(
        distributed, key=lambda item: item["interest_score"], reverse=True
    )[:1000]


def cluster_interest_anchors(anchors):
    ordered = sorted(anchors, key=lambda item: item["time"])
    clusters = []
    for anchor in ordered:
        if (
            not clusters
            or anchor["time"] - clusters[-1]["last_time"] > ANCHOR_CLUSTER_DISTANCE_SECONDS
            or anchor["time"] - clusters[-1]["start"] > MAX_ANCHOR_CLUSTER_SPAN_SECONDS
        ):
            clusters.append({
                "start": anchor["start"], "end": anchor["end"],
                "last_time": anchor["time"], "anchors": [],
            })
        cluster = clusters[-1]
        cluster["anchors"].append(anchor)
        cluster["start"] = min(cluster["start"], anchor["start"])
        cluster["end"] = max(cluster["end"], anchor["end"])
        cluster["last_time"] = anchor["time"]

    result = []
    for index, cluster in enumerate(clusters, start=1):
        peak = max(cluster["anchors"], key=lambda item: item["interest_score"])
        total = sum(item["interest_score"] for item in cluster["anchors"])
        modalities = sorted({m for item in cluster["anchors"] for m in item["modalities"]})
        cluster_score = min(
            2.0,
            peak["interest_score"] + min(0.55, (total - peak["interest_score"]) * 0.18)
            + (0.1 if len(modalities) > 1 else 0.0),
        )
        result.append({
            "cluster_id": index,
            "start": round(cluster["start"], 3),
            "end": round(cluster["end"], 3),
            "peak_time": round(peak["time"], 3),
            "peak_segment_index": peak.get("segment_index"),
            "interest_score": round(cluster_score, 4),
            "anchor_count": len(cluster["anchors"]),
            "modalities": modalities,
            "anchor_times": [item["time"] for item in cluster["anchors"]],
        })
    return sorted(result, key=lambda item: item["interest_score"], reverse=True)


def select_visual_regions(clusters, source_duration, budget_ratio=VISUAL_BUDGET_RATIO):
    desired = min(MAX_VISUAL_REGIONS, max(20, int(math.ceil(source_duration / 300.0)) * 2))
    budget = max(MAX_REGION_SECONDS, source_duration * budget_ratio)
    capacity = max(1, min(desired, int(budget // 40.0)))
    bucket_span = max(1.0, source_duration / capacity)
    best_by_bucket = {}
    for cluster in clusters:
        bucket = min(capacity - 1, int(float(cluster["peak_time"]) // bucket_span))
        current = best_by_bucket.get(bucket)
        if current is None or cluster["interest_score"] > current["interest_score"]:
            best_by_bucket[bucket] = cluster
    distributed = list(best_by_bucket.values())
    distributed_ids = {item["cluster_id"] for item in distributed}
    ordered_clusters = sorted(
        distributed, key=lambda item: item["interest_score"], reverse=True
    ) + [item for item in clusters if item["cluster_id"] not in distributed_ids]
    selected = []
    used_seconds = 0.0
    for cluster in ordered_clusters:
        peak = float(cluster["peak_time"])
        start = max(0.0, min(float(cluster["start"]) - 8.0, peak - REGION_LEAD_SECONDS))
        end = min(source_duration, max(float(cluster["end"]) + 18.0, peak + REGION_TRAIL_SECONDS))
        if end - start > MAX_REGION_SECONDS:
            start = max(0.0, peak - REGION_LEAD_SECONDS)
            end = min(source_duration, start + MAX_REGION_SECONDS)
        duration = end - start
        if duration < 12.0 or used_seconds + duration > budget:
            continue
        if any(abs(peak - item["peak_time"]) < 18.0 for item in selected):
            continue
        selected.append({
            "region_id": len(selected) + 1,
            "cluster_id": cluster["cluster_id"],
            "start": round(start, 3), "end": round(end, 3),
            "duration": round(duration, 3), "peak_time": round(peak, 3),
            "interest_score": cluster["interest_score"],
            "modalities": cluster["modalities"],
        })
        used_seconds += duration
        if len(selected) >= desired:
            break
    return selected


def merge_visual_decode_regions(regions):
    merged = []
    for region in sorted(regions, key=lambda item: item["start"]):
        if merged and region["start"] <= merged[-1]["end"] + 3.0:
            merged[-1]["end"] = max(merged[-1]["end"], region["end"])
            merged[-1]["region_ids"].append(region["region_id"])
        else:
            merged.append({
                "start": region["start"], "end": region["end"],
                "region_ids": [region["region_id"]],
            })
    for item in merged:
        item["duration"] = round(item["end"] - item["start"], 3)
    return merged


def build_event_timeline(segments, audio_events, visual_events):
    weighted = []
    for index, segment in enumerate(segments):
        interest = _text_interest(segment["text"])
        weighted.append({
            "time": segment["start"], "kind": "speech", "weight": interest,
            "segment_index": index, "text_interest": interest,
        })
    audio_weights = {
        "energy_spike": 0.82, "speech_emphasis": 0.68,
        "laughter_like": 0.86, "silence": 0.2,
    }
    for event in audio_events:
        weighted.append({
            "time": float(event["time"]), "kind": "audio",
            "weight": audio_weights.get(event.get("type"), 0.35),
            "event_type": event.get("type"),
        })
    for event in visual_events:
        weight = max(
            0.35,
            float(event.get("motion_score", 0)) * 0.72,
            float(event.get("visual_change", 0)) * 0.82,
            0.78 if event.get("scene_change") else 0,
        )
        weighted.append({
            "time": float(event["time"]), "kind": "visual",
            "weight": min(1.0, weight),
        })
    weighted.sort(key=lambda item: item["time"])

    clusters = []
    for event in weighted:
        if not clusters or event["time"] - clusters[-1]["time"] > 3.0:
            clusters.append({
                "time": event["time"], "last_time": event["time"],
                "score": 0.0, "modalities": set(), "events": [],
            })
        cluster = clusters[-1]
        cluster["last_time"] = event["time"]
        cluster["events"].append(event)
        cluster["modalities"].add(event["kind"])
        cluster["score"] += float(event["weight"])

    timeline = []
    for cluster in clusters:
        modality_bonus = 0.28 * max(0, len(cluster["modalities"]) - 1)
        weighted_time = sum(
            event["time"] * event["weight"] for event in cluster["events"]
        ) / max(sum(event["weight"] for event in cluster["events"]), 1e-6)
        timeline.append({
            "time": round(weighted_time, 2),
            "end": round(cluster["last_time"], 2),
            "score": round(cluster["score"] + modality_bonus, 4),
            "modalities": sorted(cluster["modalities"]),
            "event_count": len(cluster["events"]),
        })
    return sorted(timeline, key=lambda item: item["score"], reverse=True)


def _dead_air_ratio(start, end, segments, audio_events):
    duration = max(0.1, end - start)
    silence_seconds = 0.0
    for event in audio_events:
        if event.get("type") != "silence":
            continue
        overlap = max(
            0.0,
            min(end, float(event.get("end", event["time"])))
            - max(start, float(event["time"])),
        )
        silence_seconds += overlap
    spoken_seconds = sum(
        max(0.0, min(end, segment["end"]) - max(start, segment["start"]))
        for segment in segments
        if segment["end"] > start and segment["start"] < end
    )
    uncovered_ratio = max(0.0, 1.0 - spoken_seconds / duration)
    return round(min(1.0, max(silence_seconds / duration, uncovered_ratio)), 4)


def _best_start_index(segments, region, anchor_index):
    candidates = []
    peak = float(region["peak_time"])
    for index, segment in enumerate(segments):
        if segment["end"] < peak - REGION_LEAD_SECONDS or segment["start"] > peak + 1.5:
            continue
        previous_gap = (
            segment["start"] - segments[index - 1]["end"] if index else 1.0
        )
        begins_sentence = index == 0 or segments[index - 1]["text"].rstrip().endswith((".", "!", "?"))
        score = (
            _text_interest(segment["text"]) * 3.0
            + min(1.5, max(0.0, previous_gap) * 1.5)
            + (1.2 if begins_sentence else 0.0)
            - abs(index - anchor_index) * 0.08
        )
        candidates.append((score, index))
    return max(candidates, default=(0.0, anchor_index), key=lambda item: item[0])[1]


def _candidate_variant(segments, region, start_index, variant, audio_events, visual_events):
    ranges = {
        "SHORT": (15.0, 25.0, 21.0),
        "STANDARD": (25.0, 40.0, 32.0),
        "EXTENDED": (35.0, 55.0, 47.0),
    }
    minimum, maximum, target = ranges[variant]
    start = float(segments[start_index]["start"])
    endpoints = []
    for end_index in range(start_index, len(segments)):
        end = float(segments[end_index]["end"])
        duration = end - start
        if duration > maximum:
            break
        if duration < minimum:
            continue
        terminal = segments[end_index]["text"].rstrip().endswith((".", "!", "?"))
        next_gap = (
            max(0.0, segments[end_index + 1]["start"] - end)
            if end_index + 1 < len(segments) else 1.0
        )
        payoff_terms = (
            "because", "that's why", "therefore", "the answer", "because of",
            "weil", "deshalb", "darum", "die antwort", "am ende",
        )
        lower = segments[end_index]["text"].lower()
        payoff = 1.6 if any(term in lower for term in payoff_terms) else 0.0
        boundary = (
            (3.0 if terminal else 0.0) + min(2.5, next_gap * 2.5) + payoff
            - abs(duration - target) * 0.12
        )
        endpoints.append((boundary, end_index))
    if not endpoints:
        return None
    _, end_index = max(endpoints, key=lambda item: item[0])
    chosen = segments[start_index:end_index + 1]
    end = float(chosen[-1]["end"])
    text = " ".join(segment["text"] for segment in chosen).strip()
    if not text:
        return None
    audio = _audio_summary(start, end, audio_events)
    visual = _visual_summary(start, end, visual_events)
    dead_air = _dead_air_ratio(start, end, chosen, audio_events)
    text_interest = max((_text_interest(segment["text"]) for segment in chosen), default=0.0)
    audio_score = min(
        1.0,
        audio["energy_spikes"] * 0.2 + audio["rapid_speech_events"] * 0.13
        + audio["speech_rate_changes"] * 0.1 + audio["laughter_like_events"] * 0.28
        + audio["peak_energy"] * 0.29,
    )
    visual_score = min(
        1.0,
        visual["scene_changes"] * 0.16 + visual["face_count_changes"] * 0.08
        + visual["motion_peak"] * 0.3 + visual["visual_change_peak"] * 0.35
        + visual["possible_zooms"] * 0.08,
    )
    active_modalities = sum(value >= 0.24 for value in (text_interest, audio_score, visual_score))
    local_score = round(
        text_interest * 42 + min(1.0, float(region["interest_score"])) * 20
        + audio_score * 18 + visual_score * 10
        + max(0, active_modalities - 1) * 5 - dead_air * 22
    )
    if text_interest < 0.32:
        local_score = min(local_score, 58)
    return {
        "start": round(start, 3), "end": round(end, 3),
        "duration": round(end - start, 3), "text": text,
        "local_multimodal_score": int(max(0, min(100, local_score))),
        "local_interest_score": round(float(region["interest_score"]), 4),
        "region_id": region["region_id"], "region_rank": region["region_id"],
        "region_type": variant,
        "anchor_types": region.get("modalities", []),
        "anchor_strengths": {
            "region_interest": round(float(region["interest_score"]), 4),
        },
        "seed_time": round(float(region["peak_time"]), 3),
        "seed_modalities": region.get("modalities", []),
        "dead_air_ratio": dead_air, "audio": audio, "visual": visual,
    }


def generate_region_candidates(segments, regions, audio_events, visual_events):
    raw = []
    for region in regions:
        peak = float(region["peak_time"])
        anchor_index = min(
            range(len(segments)),
            key=lambda index: abs(
                (segments[index]["start"] + segments[index]["end"]) / 2 - peak
            ),
        )
        start_index = _best_start_index(segments, region, anchor_index)
        for variant in ("SHORT", "STANDARD", "EXTENDED"):
            candidate = _candidate_variant(
                segments, region, start_index, variant, audio_events, visual_events
            )
            if candidate is not None:
                raw.append(candidate)
    raw.sort(key=lambda item: item["local_multimodal_score"], reverse=True)

    deduplicated = []
    for candidate in raw:
        duplicate = False
        candidate_words = set(re.findall(r"\w+", candidate["text"].lower()))
        for kept in deduplicated:
            overlap = _overlap_ratio(candidate, kept)
            kept_words = set(re.findall(r"\w+", kept["text"].lower()))
            union = candidate_words | kept_words
            similarity = len(candidate_words & kept_words) / max(1, len(union))
            structurally_distinct = (
                candidate["region_type"] != kept["region_type"]
                and abs(candidate["duration"] - kept["duration"]) >= 9.0
                and abs(candidate["end"] - kept["end"]) >= 6.0
            )
            if overlap >= 0.84 and similarity >= 0.72 and not structurally_distinct:
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(candidate)

    selected = []
    for candidate in deduplicated:
        conflicts = [kept for kept in selected if _overlap_ratio(candidate, kept) >= 0.68]
        if conflicts and not all(
            candidate["region_type"] != kept["region_type"]
            and abs(candidate["duration"] - kept["duration"]) >= 10.0
            and abs(candidate["end"] - kept["end"]) >= 7.0
            for kept in conflicts
        ):
            continue
        selected.append(candidate)
        if len(selected) >= MAX_SELECTED_CANDIDATES:
            break
    return raw, deduplicated, selected


def generate_candidates(segments, timeline, audio_events, visual_events, source_duration):
    if not segments:
        return [], []
    seeds = list(timeline)
    existing_times = [seed["time"] for seed in seeds]
    for index, segment in sorted(
        enumerate(segments), key=lambda item: _text_interest(item[1]["text"]), reverse=True
    ):
        center = (segment["start"] + segment["end"]) / 2
        if all(abs(center - time_value) > 2.5 for time_value in existing_times):
            seeds.append({
                "time": round(center, 2), "end": round(segment["end"], 2),
                "score": round(_text_interest(segment["text"]), 4),
                "modalities": ["speech"], "event_count": 1,
            })
            existing_times.append(center)
        if len(seeds) >= 80:
            break
    seeds.sort(key=lambda item: item["score"], reverse=True)

    raw = []
    for seed in seeds:
        candidate = _candidate_around_seed(
            segments, seed, audio_events, visual_events, source_duration
        )
        if candidate is None:
            continue
        if any(_overlap_ratio(candidate, kept) >= 0.9 for kept in raw):
            continue
        raw.append(candidate)
        if len(raw) >= MAX_RAW_CANDIDATES:
            break
    raw.sort(key=lambda item: item["local_multimodal_score"], reverse=True)

    selected = []
    for candidate in raw:
        if any(_overlap_ratio(candidate, kept) >= 0.6 for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= MAX_SELECTED_CANDIDATES:
            break
    return raw, selected


def _candidate_around_seed(segments, seed, audio_events, visual_events, source_duration):
    seed_time = float(seed["time"])
    anchor_index = min(
        range(len(segments)),
        key=lambda index: abs(
            ((segments[index]["start"] + segments[index]["end"]) / 2) - seed_time
        ),
    )
    start_index = anchor_index
    while start_index > 0 and segments[start_index]["start"] - seed_time > -2.5:
        previous = segments[start_index - 1]
        gap = segments[start_index]["start"] - previous["end"]
        if gap > 0.8:
            break
        start_index -= 1
    start = max(0.0, segments[start_index]["start"])

    endpoints = []
    for end_index in range(start_index, len(segments)):
        end = min(float(source_duration), segments[end_index]["end"])
        duration = end - start
        if duration > MAX_CLIP_SECONDS:
            break
        if duration < MIN_CLIP_SECONDS:
            continue
        terminal = segments[end_index]["text"].rstrip().endswith((".", "!", "?"))
        next_gap = (
            max(0.0, segments[end_index + 1]["start"] - end)
            if end_index + 1 < len(segments) else 1.0
        )
        nearby_visual_boundary = any(
            abs(float(event["time"]) - end) <= 1.2 and event.get("scene_change")
            for event in visual_events
        )
        boundary_score = (
            (4.0 if terminal else 0.0)
            + min(3.0, next_gap * 3.0)
            + (2.0 if nearby_visual_boundary else 0.0)
            - abs(duration - 28.0) * 0.065
        )
        endpoints.append((boundary_score, end_index))
    if not endpoints:
        return None
    _, end_index = max(endpoints, key=lambda item: item[0])
    chosen = segments[start_index:end_index + 1]
    end = chosen[-1]["end"]
    duration = end - start
    text = " ".join(segment["text"] for segment in chosen).strip()

    audio = _audio_summary(start, end, audio_events)
    visual = _visual_summary(start, end, visual_events)
    text_score = max((_text_interest(segment["text"]) for segment in chosen), default=0.0)
    audio_score = min(
        1.0,
        audio["energy_spikes"] * 0.22 + audio["rapid_speech_events"] * 0.13
        + audio["laughter_like_events"] * 0.3 + audio["peak_energy"] * 0.35,
    )
    visual_score = min(
        1.0,
        visual["scene_changes"] * 0.18 + visual["face_count_changes"] * 0.12
        + visual["motion_peak"] * 0.3 + visual["visual_change_peak"] * 0.4,
    )
    active_modalities = sum(value > 0.18 for value in (text_score, audio_score, visual_score))
    multimodal_bonus = max(0, active_modalities - 1) * 6
    local_score = min(
        100,
        round(text_score * 42 + audio_score * 25 + visual_score * 25 + multimodal_bonus),
    )
    return {
        "start": round(start, 3), "end": round(end, 3),
        "duration": round(duration, 3), "text": text,
        "local_multimodal_score": int(local_score),
        "seed_time": round(seed_time, 3),
        "seed_modalities": seed.get("modalities", []),
        "audio": audio, "visual": visual,
    }


def _audio_summary(start, end, events):
    selected = [event for event in events if start <= float(event["time"]) <= end]
    return {
        "energy_spikes": sum(event.get("type") == "energy_spike" for event in selected),
        "peak_energy": round(max((float(event.get("peak_energy", event.get("audio_energy", 0))) for event in selected), default=0.0), 4),
        "rapid_speech_events": sum(bool(event.get("rapid_speech")) for event in selected),
        "speech_rate_changes": sum(float(event.get("rate_change", 0)) >= 0.45 for event in selected),
        "laughter_like_events": sum(event.get("type") == "laughter_like" for event in selected),
        "long_silences": sum(event.get("type") == "silence" for event in selected),
    }


def _visual_summary(start, end, events):
    selected = [event for event in events if start <= float(event["time"]) <= end]
    return {
        "scene_changes": sum(bool(event.get("scene_change")) for event in selected),
        "motion_peak": round(max((float(event.get("motion_score", 0)) for event in selected), default=0.0), 4),
        "visual_change_peak": round(max((float(event.get("visual_change", 0)) for event in selected), default=0.0), 4),
        "possible_zooms": sum(bool(event.get("possible_zoom")) for event in selected),
        "face_count_changes": sum(bool(event.get("face_count_change")) for event in selected),
        "faces_present": any(int(event.get("face_count", 0)) > 0 for event in selected),
    }


def _overlap_ratio(first, second):
    overlap = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    return overlap / max(1.0, min(first["duration"], second["duration"]))


def analyze_project(project_dir):
    project_dir = Path(project_dir).resolve()
    output_path = project_dir / "analysis_features.json"
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        return _analyze_project(project_dir, output_path, started, started_at)
    except Exception as error:
        failure = {
            "analysis_version": ANALYSIS_VERSION,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "diagnostics": {
                "total_elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        }
        _atomic_json(output_path, failure)
        raise


def _analyze_project(project_dir, output_path, started, started_at):
    transcript_path = project_dir / "transcript.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = _clean_segments(transcript)
    source = _find_source(project_dir, transcript)
    if source is None:
        raise FileNotFoundError(f"Source video not found in {project_dir}")
    duration = float(transcript.get("duration") or 0)
    if duration <= 0 and segments:
        duration = max(segment["end"] for segment in segments)
    fingerprint = _analysis_fingerprint(source, transcript_path)
    cached = _read_json(output_path)
    required_cache_keys = (
        "global_audio_events", "interest_anchors", "clustered_regions",
        "regions_selected_for_visual_analysis", "visual_events",
        "selected_candidates", "performance",
    )
    if (
        cached
        and cached.get("success") is True
        and _cache_matches(cached, fingerprint, required_cache_keys)
    ):
        print("MULTIMODAL ANALYSIS CACHE HIT", flush=True)
        cached, semantic_changed = update_semantic_features(cached, transcript)
        if semantic_changed:
            _atomic_json(output_path, cached)
        return cached

    timings = {}
    two_stage = duration > TWO_STAGE_THRESHOLD_SECONDS
    if two_stage:
        print("TWO STAGE ANALYSIS ACTIVE", flush=True)

    stage_a_path = project_dir / "analysis_stage_a.json"
    stage_a_cache = _read_json(stage_a_path)
    if _cache_matches(
        stage_a_cache,
        fingerprint,
        ("audio_energy_samples", "global_audio_events", "interest_anchors", "clustered_regions"),
    ):
        energy_samples = stage_a_cache["audio_energy_samples"]
        audio_events = stage_a_cache["global_audio_events"]
        audio_diagnostics = stage_a_cache.get("audio_diagnostics", {"available": True, "cached": True})
        anchors = stage_a_cache["interest_anchors"]
        clusters = stage_a_cache["clustered_regions"]
        timings["audio_analysis_seconds"] = 0.0
        timings["anchor_generation_seconds"] = 0.0
        print("STAGE A CACHE HIT", flush=True)
    else:
        audio_started = time.perf_counter()
        print("MULTIMODAL ANALYSIS: audio", flush=True)
        try:
            energy_samples, audio_events, audio_diagnostics = analyze_audio(
                source, transcript, project_dir
            )
        except Exception as audio_error:
            energy_samples, audio_events = [], []
            audio_diagnostics = {
                "available": False,
                "reason": f"{type(audio_error).__name__}: {audio_error}",
            }
            if two_stage:
                print(
                    "TWO STAGE ANALYSIS FALLBACK: global audio analysis failed: "
                    f"{type(audio_error).__name__}: {audio_error}",
                    flush=True,
                )
        timings["audio_analysis_seconds"] = time.perf_counter() - audio_started
        print(f"AUDIO EVENTS: {len(audio_events)}", flush=True)
        anchor_started = time.perf_counter()
        try:
            anchors = generate_interest_anchors(segments, energy_samples, audio_events)
            clusters = cluster_interest_anchors(anchors)
        except Exception as anchor_error:
            if two_stage:
                print(
                    "TWO STAGE ANALYSIS FALLBACK: interest anchor generation failed: "
                    f"{type(anchor_error).__name__}: {anchor_error}",
                    flush=True,
                )
            anchors = [
                {
                    "time": round((segment["start"] + segment["end"]) / 2, 3),
                    "start": segment["start"], "end": segment["end"],
                    "segment_index": index, "text": segment["text"],
                    "interest_score": _text_interest(segment["text"]),
                    "modalities": ["text"],
                    "components": {"text_interest": _text_interest(segment["text"])},
                }
                for index, segment in sorted(
                    enumerate(segments),
                    key=lambda item: _text_interest(item[1]["text"]),
                    reverse=True,
                )[:200]
            ]
            clusters = cluster_interest_anchors(anchors)
        timings["anchor_generation_seconds"] = time.perf_counter() - anchor_started
        _atomic_json(stage_a_path, {
            "analysis_version": ANALYSIS_VERSION,
            "fingerprint": fingerprint,
            "audio_energy_samples": energy_samples,
            "global_audio_events": audio_events,
            "audio_diagnostics": audio_diagnostics,
            "interest_anchors": anchors,
            "clustered_regions": clusters,
        })
    print(f"INTEREST ANCHORS: {len(anchors)}", flush=True)

    visual_regions = select_visual_regions(
        clusters, duration, VISUAL_BUDGET_RATIO if two_stage else 1.0
    )
    decode_regions = merge_visual_decode_regions(visual_regions)
    visual_seconds = (
        sum(region["duration"] for region in decode_regions) if two_stage else duration
    )
    print(f"VISUAL REGIONS: {len(visual_regions)}", flush=True)

    visual_started = time.perf_counter()
    print("MULTIMODAL ANALYSIS: video", flush=True)
    if two_stage and not decode_regions:
        visual_events = []
        video_diagnostics = {
            "available": False,
            "reason": "no visual regions selected",
        }
    else:
        try:
            visual_events, video_diagnostics = analyze_video(
                source,
                duration,
                audio_events,
                regions=decode_regions if two_stage else None,
            )
        except Exception as visual_error:
            visual_events = []
            video_diagnostics = {
                "available": False,
                "reason": f"{type(visual_error).__name__}: {visual_error}",
            }
            if two_stage:
                print(
                    "TWO STAGE ANALYSIS FALLBACK: visual region analysis failed: "
                    f"{type(visual_error).__name__}: {visual_error}",
                    flush=True,
                )
    timings["visual_region_analysis_seconds"] = time.perf_counter() - visual_started
    if two_stage and not video_diagnostics.get("available", False):
        print(
            "TWO STAGE ANALYSIS FALLBACK: visual features unavailable: "
            f"{video_diagnostics.get('reason', 'unknown reason')}",
            flush=True,
        )
    print(f"VIDEO EVENTS: {len(visual_events)}", flush=True)
    print(
        "VISUAL ANALYZED: "
        f"{int(visual_seconds // 60)}m {int(visual_seconds % 60)}s / "
        f"SOURCE {int(duration // 60)}m {int(duration % 60)}s",
        flush=True,
    )

    candidate_started = time.perf_counter()
    if segments and visual_regions:
        try:
            raw_candidates, deduplicated_candidates, selected_candidates = (
                generate_region_candidates(
                    segments, visual_regions, audio_events, visual_events
                )
            )
        except Exception as candidate_error:
            print(
                "TWO STAGE ANALYSIS FALLBACK: region candidate generation failed: "
                f"{type(candidate_error).__name__}: {candidate_error}",
                flush=True,
            )
            timeline_fallback = build_event_timeline(segments, audio_events, visual_events)
            raw_candidates, selected_candidates = generate_candidates(
                segments, timeline_fallback, audio_events, visual_events, duration
            )
            deduplicated_candidates = raw_candidates
    else:
        timeline_fallback = build_event_timeline(segments, audio_events, visual_events)
        raw_candidates, selected_candidates = generate_candidates(
            segments, timeline_fallback, audio_events, visual_events, duration
        )
        deduplicated_candidates = raw_candidates
        if two_stage:
            print(
                "TWO STAGE ANALYSIS FALLBACK: no usable visual regions; "
                "using transcript/audio candidates",
                flush=True,
            )
    timings["candidate_generation_seconds"] = time.perf_counter() - candidate_started
    timings["analysis_total_seconds"] = time.perf_counter() - started
    visual_ratio = visual_seconds / duration if duration > 0 else 0.0
    output = {
        "analysis_version": ANALYSIS_VERSION,
        "success": True,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(source),
        "fingerprint": fingerprint,
        "duration": round(duration, 3),
        "total_source_duration": round(duration, 3),
        "config": {
            "two_stage_threshold_seconds": TWO_STAGE_THRESHOLD_SECONDS,
            "two_stage_active": two_stage,
            "anchor_cluster_distance_seconds": ANCHOR_CLUSTER_DISTANCE_SECONDS,
            "max_visual_regions": MAX_VISUAL_REGIONS,
            "visual_budget_ratio": VISUAL_BUDGET_RATIO,
            "region_lead_seconds": REGION_LEAD_SECONDS,
            "region_trail_seconds": REGION_TRAIL_SECONDS,
            "video_base_sample_fps": 1.0,
            "video_event_sample_fps": 2.0,
            "clip_duration_range": [MIN_CLIP_SECONDS, MAX_CLIP_SECONDS],
            "max_raw_candidates": MAX_RAW_CANDIDATES,
            "max_gpt_candidates": MAX_SELECTED_CANDIDATES,
        },
        "diagnostics": {
            "audio": audio_diagnostics,
            "video": video_diagnostics,
            "total_elapsed_seconds": round(time.perf_counter() - started, 2),
        },
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "performance": {
            "source_duration_seconds": round(duration, 3),
            "visual_analyzed_seconds": round(visual_seconds, 3),
            "visual_analysis_ratio": round(visual_ratio, 4),
        },
        "audio_energy_samples": energy_samples,
        "audio_events": audio_events,
        "global_audio_events": audio_events,
        "interest_anchors": anchors,
        "clustered_regions": clusters,
        "regions_selected_for_visual_analysis": visual_regions,
        "visual_decode_regions": decode_regions,
        "visual_events": visual_events,
        "event_timeline": clusters,
        "raw_candidate_variants": raw_candidates,
        "deduplicated_candidates": deduplicated_candidates,
        "candidates_sent_to_ai": selected_candidates,
        "raw_candidates": raw_candidates,
        "selected_candidates": selected_candidates,
    }
    output, _ = update_semantic_features(output, transcript)
    _atomic_json(output_path, output)
    _update_timings(project_dir, timings)
    print(f"RAW MULTIMODAL CANDIDATES: {len(raw_candidates)}", flush=True)
    print(f"SELECTED FOR GPT: {len(output['selected_candidates'])}", flush=True)
    print(f"WROTE: {output_path}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    args = parser.parse_args()
    analyze_project(args.project_dir)


if __name__ == "__main__":
    main()
