import os
import re
import math
import random
import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import ujson as json
import yaml
import transformers
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000

logger = logging.getLogger(__name__)

###############################################################################
# Basic helper logic: rounding, resizing, frames, etc.
###############################################################################
def round_by_factor(number: float, factor: int) -> int:
    """Round 'number' to the nearest integer multiple of 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: float, factor: int) -> int:
    """Ceil 'number' to the nearest integer multiple of 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: float, factor: int) -> int:
    """Floor 'number' to the nearest integer multiple of 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: float,
) -> Tuple[int, int]:
    """
    Rescale (height, width) so that:
      - aspect ratio <= max_ratio,
      - total area in [min_pixels, max_pixels],
      - each dimension is multiple of factor.
    """
    ratio = max(height, width) / min(height, width)
    if ratio > max_ratio:
        raise ValueError(f"Aspect ratio {ratio:.2f} > {max_ratio}")

    h_ = max(factor, round_by_factor(height, factor))
    w_ = max(factor, round_by_factor(width, factor))
    area = h_ * w_

    if area > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        h_ = floor_by_factor(height / scale, factor)
        w_ = floor_by_factor(width / scale, factor)
    elif area < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h_ = ceil_by_factor(height * scale, factor)
        w_ = ceil_by_factor(width * scale, factor)
    return h_, w_

def _smart_nframes(
    config: Dict[str, Any],
    total_frames: int,
    video_fps: float,
    frame_factor: int,
    default_fps: float,
    fps_min_frames: int,
    fps_max_frames: int
) -> int:
    """
    Decide how many frames to pick from a video based on either:
       - 'nframes' in config
       - or 'fps' in config (or default_fps if none specified).
    Result is clamped to [fps_min_frames, fps_max_frames], 
    and must be multiple of 'frame_factor'.
    """
    if "nframes" in config and "fps" in config:
        raise ValueError("Provide only one of `fps` or `nframes` in config.")

    if "nframes" in config:
        nframes = round_by_factor(config["nframes"], frame_factor)
    else:
        user_fps = config.get("fps", default_fps)
        minf = ceil_by_factor(config.get("min_frames", fps_min_frames), frame_factor)
        maxf = floor_by_factor(config.get("max_frames", min(fps_max_frames, total_frames)), frame_factor)
        val = total_frames / video_fps * user_fps
        val = min(max(val, minf), maxf)
        nframes = round_by_factor(val, frame_factor)

    if not (frame_factor <= nframes <= total_frames):
        raise ValueError(f"Invalid nframes={nframes}, out of range.")
    return int(nframes)

def _read_video_torchvision(
    config: Dict[str, Any],
    frame_factor: int,
    default_fps: float,
    fps_min_frames: int,
    fps_max_frames: int
) -> torch.Tensor:
    """
    Use torchvision.io.read_video to read and return a TCHW video tensor.
    """
    path = config["video"]
    vid, _, info = io.read_video(
        path,
        start_pts=config.get("video_start", 0.0),
        end_pts=config.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames = vid.size(0)
    video_fps = info["video_fps"]
    nframes = _smart_nframes(config, total_frames, video_fps, frame_factor, default_fps, fps_min_frames, fps_max_frames)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    return vid[idx]

def _read_video_decord(
    config: Dict[str, Any],
    frame_factor: int,
    default_fps: float,
    fps_min_frames: int,
    fps_max_frames: int
) -> torch.Tensor:
    """
    Use decord to read and return a TCHW video tensor.
    """
    import decord
    path = config["video"]
    vr = decord.VideoReader(path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    nframes = _smart_nframes(config, total_frames, video_fps, frame_factor, default_fps, fps_min_frames, fps_max_frames)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    arr = vr.get_batch(idx).asnumpy()  # T,H,W,C
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # -> T,C,H,W

VIDEO_READERS = {
    "torchvision": _read_video_torchvision,
    "decord": _read_video_decord,
}

def pick_video_reader() -> str:
    """Pick decord if installed, otherwise torchvision."""
    try:
        import importlib.util
        if importlib.util.find_spec("decord") is not None:
            return "decord"
    except:
        pass
    return "torchvision"

###############################################################################
# Multimodal fetch_image / fetch_video
###############################################################################
def _fetch_image(
    config: Dict[str, Any],
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: float
) -> Image.Image:
    """
    Load a single image (from local path, URL, or base64) 
    and resize it via 'smart_resize' constraints.
    """
    source = config.get("image") or config.get("image_url")
    if not source:
        raise ValueError("Must provide either 'image' or 'image_url' in config.")

    # Load
    if isinstance(source, Image.Image):
        pil_img = source
    elif isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://"):
            import requests
            pil_img = Image.open(requests.get(source, stream=True).raw)
        elif source.startswith("file://"):
            pil_img = Image.open(source[7:])
        elif source.startswith("data:image"):
            # base64 data
            if "base64," in source:
                _, b64_data = source.split("base64,", 1)
                raw = base64.b64decode(b64_data)
                pil_img = Image.open(BytesIO(raw))
            else:
                raise ValueError("Invalid base64 image data.")
        else:
            # local path
            pil_img = Image.open(source)
    else:
        raise ValueError(f"Unsupported type for 'image': {type(source)}")

    pil_img = pil_img.convert("RGB")

    # Resize
    if "resized_height" in config and "resized_width" in config:
        rh, rw = smart_resize(config["resized_height"], config["resized_width"], image_factor, min_pixels, max_pixels, max_ratio)
    else:
        # infer dims from the image
        w, h = pil_img.size
        local_min = config.get("min_pixels", min_pixels)
        local_max = config.get("max_pixels", max_pixels)
        rh, rw = smart_resize(h, w, image_factor, local_min, local_max, max_ratio)

    # Return the resized image
    return pil_img.resize((rw, rh))

def _fetch_video(
    config: Dict[str, Any],
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: float,
    video_total_pixels: int,
    frame_factor: int,
    default_fps: float,
    fps_min_frames: int,
    fps_max_frames: int
) -> Union[torch.Tensor, List[Image.Image]]:
    """
    If config['video'] is a str => read entire video => TCHW tensor,
    If config['video'] is a list => treat them as frame paths => list of PIL Images.
    """
    val = config["video"]
    if isinstance(val, str):
        # Single video path
        backend = pick_video_reader()
        fn = VIDEO_READERS[backend]
        vid_tensor = fn(config, frame_factor, default_fps, fps_min_frames, fps_max_frames)
        # shape => T, C, oh, ow
        t, c, oh, ow = vid_tensor.shape
        local_min = config.get("min_pixels", min_pixels)
        local_max = config.get("max_pixels", max_pixels)
        local_total = config.get("total_pixels", video_total_pixels)
        guess_max = max(min(local_max, local_total / t * frame_factor), int(local_min * 1.05))

        if "resized_height" in config and "resized_width" in config:
            rh, rw = smart_resize(config["resized_height"], config["resized_width"], image_factor, local_min, guess_max, max_ratio)
        else:
            rh, rw = smart_resize(oh, ow, image_factor, local_min, guess_max, max_ratio)

        # Resize frames
        vid_tensor = transforms.functional.resize(
            vid_tensor,
            [rh, rw],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ).float()
        return vid_tensor

    elif isinstance(val, list):
        # List of frame paths
        frames = []
        meta = dict(config)
        meta.pop("video", None)
        for fp in val:
            meta["image"] = fp
            frames.append(_fetch_image(meta, image_factor, min_pixels, max_pixels, max_ratio))

        # Possibly pad frames to multiple of frame_factor
        needed = ceil_by_factor(len(frames), frame_factor)
        if len(frames) < needed and len(frames) > 0:
            frames += [frames[-1]] * (needed - len(frames))
        return frames

    else:
        raise ValueError(f"'video' must be a str or list, got {type(val)}")


def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    return tokenizer(prompt, return_tensors=return_tensors).input_ids[0]