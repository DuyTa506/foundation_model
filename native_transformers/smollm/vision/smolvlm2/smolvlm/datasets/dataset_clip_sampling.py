import os
import math
import random
import time
import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge("torch")
from num2words import num2words
import datetime
import re
import torch
import numpy as np
import transformers
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from smolvlm.constants import (
    IGNORE_INDEX,
    DATA_IMAGE_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DATA_VIDEO_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)
from smolvlm.train.args import DataArguments, TrainingArguments, ModelArguments
from smolvlm.utils import mprint

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000


##############################################################################
# helper functions
##############################################################################


DEFAULT_SYSTEM_MESSAGE = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
# DEFAULT_VIDEO_INTRO = "Here are some frames sampled from a video:"
DEFAULT_VIDEO_INTRO = (
    "You are provided the following {clip_count} clips sampled "
    "from a {video_duration} [H:MM:SS] video. The clips:\n"
)
DEFAULT_IMAGE_INTRO = "Here are some images:\n"
DEFAULT_MEDIA_OUTTRO = "\n\n"
FRAME_TIMESTAMP_MESSAGE_BASE = "\nClip from "
FRAME_TIMESTAMP_MESSAGE = FRAME_TIMESTAMP_MESSAGE_BASE + "{clip_ts}:"



def sample_clip_indices(frames_per_clip: int, video_duration: float, sampling_fps: float, video_fps: float, max_clips: int):
    # Total number of frames in the video and effective clip duration.
    tot_frames = int(video_duration * video_fps)
    clip_dur = frames_per_clip / sampling_fps
    
    # Determine the number of possible clips, then cap at max_clips.
    possible = math.ceil(video_duration / clip_dur)
    n_clips = min(max_clips, possible)
    
    # r > 0.5 means we can sample without repeating frames.
    r = video_fps / sampling_fps
    clips, timestamps = [], []
    
    def pad(arr, target, pad_val):
        return arr if arr.size >= target else np.concatenate((arr, np.full(target - arr.size, pad_val, dtype=np.int64)))
    
    if r > 0.5:
        step = max(1, int(round(video_fps / sampling_fps)))
        seg_frames = int(round(frames_per_clip * step))
        valid = min(seg_frames, tot_frames)
        # If more clips than a single continuous segment, space them out.
        gap = ((tot_frames - seg_frames) // (n_clips - 1)) if n_clips > 1 and tot_frames > seg_frames else 0
        part = tot_frames // n_clips
        for i in range(n_clips):
            if part > seg_frames:
                off = (part - seg_frames) // 2
                idx = np.arange(off, off + seg_frames, step)
                idx = np.clip(idx, 0, part - 1) + i * part
            else:
                idx = np.arange(0, valid, step)
                idx = pad(idx, frames_per_clip, valid - 1)
                idx = np.clip(idx, 0, valid - 1) + i * gap
            idx = pad(idx, frames_per_clip, idx[-1] if idx.size else 0)
            clips.append(idx.astype(np.int64))
            timestamps.append((idx[0] / video_fps, idx[-1] / video_fps))
    else:
        reps = int(math.ceil(1 / r))
        seg_frames = int(round(frames_per_clip * r))
        base = np.repeat(np.arange(seg_frames), reps)
        gap = ((tot_frames - seg_frames) // (n_clips - 1)) if n_clips > 1 and tot_frames > seg_frames else 0
        valid = min(seg_frames, tot_frames)
        base = pad(base, frames_per_clip, valid - 1)
        for i in range(n_clips):
            idx = np.clip(base, 0, valid - 1).astype(np.int64) + i * gap
            clips.append(idx)
            timestamps.append((idx[0] / video_fps, idx[-1] / video_fps))
    
    all_indices = np.concatenate(clips).tolist()
    return all_indices, timestamps


def sample_clip_indices(
    frames_per_clip: int,
    video_duration: float,
    sampling_fps: float,
    video_fps: float,
    max_clips: int
):
    """
    This function returns *up to* (frames_per_clip * max_clips) frame indices in total,
    grouped into some number of clips <= max_clips. Each clip is size frames_per_clip,
    and we also return a (start_sec, end_sec) for each clip.

    If the video is very short, we produce fewer clips rather than replicating frames.

    Returns:
      all_indices (List[int]): A flattened list of frame indices, grouped into
        (frames_per_clip * actual_clips).
      timestamps  (List[Tuple[float, float]]): A list of clip-level (start_sec, end_sec).
    """
    
    tot_frames = int(round(video_duration * video_fps))
    if tot_frames <= 0:
        return [], []

    # At most, how many full clips can we form without replicating frames?
    max_full_clips = tot_frames // frames_per_clip
    # The actual number of clips is the minimum of that and the requested max_clips
    actual_clips = min(max_full_clips, max_clips)
    # If there's room for at least 1 clip
    if actual_clips < 1:
        # In an extreme scenario, we have fewer frames than frames_per_clip
        # We'll just produce 1 partial clip. Or you can choose to return empty.
        actual_clips = 1 if tot_frames > 0 else 0

    if actual_clips == 0:
        # No frames => no clips
        return [], []

    desired_frames = frames_per_clip * actual_clips
    threshold_time = desired_frames / float(sampling_fps) if sampling_fps > 0 else 0.0

    # Decide whether to do short-video step-based approach or uniform approach
    # based on video_duration <= threshold_time
    if sampling_fps <= 0:
        # if sampling_fps is 0 or negative, fallback to uniform
        sampling_short = False
    else:
        sampling_short = (video_duration <= threshold_time)

    if sampling_short:
        # STEP-BASED approach
        step = video_fps / float(sampling_fps) if sampling_fps > 0 else 1.0
        step_indices = []
        idx = 0.0
        while True:
            frame_idx = int(round(idx))
            if frame_idx >= tot_frames:
                break
            step_indices.append(frame_idx)
            idx += step

        # If we got more frames than needed, slice them
        if len(step_indices) > desired_frames:
            step_indices = step_indices[:desired_frames]
        # If we got fewer, don't replicate. We'll just produce fewer final frames.
        indices = step_indices

    else:
        # UNIFORM approach
        if tot_frames == 1:
            # If there's only 1 frame, replicate? Or produce 1?
            # We'll produce just 1 in total. That means actual_clips=1, frames_per_clip=1
            indices = [0]
        else:
            # linearly space exactly 'desired_frames' from [0 .. tot_frames-1]
            # but if the video is short, we might not want to forcibly replicate
            # => We'll do the linear spacing, no replication beyond tot_frames-1
            lin = np.linspace(0, tot_frames - 1, desired_frames, dtype=np.float32)
            indices = np.round(lin).astype(int).tolist()

    # Now we have a raw list of frame indices (no duplicates forcibly appended).
    # Possibly we have fewer frames than desired_frames. Let's recalc:
    actual_frames = len(indices)
    # Recompute how many complete clips we can form
    final_clips = actual_frames // frames_per_clip
    if final_clips < 1:
        # If we can't form even 1 clip, no point returning partial
        return [], []

    # We'll keep only what forms full clips
    final_needed = final_clips * frames_per_clip
    indices = indices[:final_needed]

    # Next, chunk into final_clips
    clip_indices = []
    timestamps = []
    offset = 0
    for _ in range(final_clips):
        chunk = indices[offset : offset + frames_per_clip]
        offset += frames_per_clip
        start_time = chunk[0] / float(video_fps)
        end_time   = chunk[-1] / float(video_fps)
        timestamps.append((start_time, end_time))
        clip_indices.append(chunk)

    # Flatten
    all_indices = sum(clip_indices, [])
    return all_indices, timestamps
    
# Video Loader
##############################################################################

def load_video(
    path: str,
    max_frames: int = 100,
    target_fps: float = 2.0,
    frames_per_clip: int =1,   
) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads a video from `path` using decord, sampling up to `max_frames` frames.
    After deduplicating indices (e.g., to handle rounding collisions), each frame
    is decoded into a PIL Image (in RGB mode). Timestamps are generated in "MM:SS" format
    based on the frame index over `native_fps`.

    Args:
        path (str): Path to the video file (e.g., MP4).
        max_frames (int): Hard cap on how many frames we ever pick in total.
        target_fps (float): Target approximate sampling rate in frames per second.
    
    Returns:
        Tuple[List[Image.Image], List[str]]:
          - A list of PIL.Image objects corresponding to each selected frame.
          - A list of parallel timestamps ("MM:SS" strings), one per selected frame.
    """
    try:
        # Use decord with single-thread and CPU context
        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    except Exception as e:
        raise RuntimeError(f"Failed to open video '{path}': {e}")

    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"Video '{path}' has 0 frames.")

    # Fallback to 30 if native_fps is None or zero
    native_fps = vr.get_avg_fps() or 30.0
    duration_seconds = total_frames / native_fps

    frames_idx, clip_times = sample_clip_indices(
        frames_per_clip=frames_per_clip,       
        video_duration=duration_seconds,
        sampling_fps=target_fps,
        video_fps=native_fps,
        max_clips=max_frames     # as an example usage
    )
    # Read frames from decord
    try:
        frames_tensor = vr.get_batch(frames_idx).cpu().numpy()  # (N, H, W, C)
    except Exception as e:
        raise RuntimeError(f"Failed to read frames from '{path}': {e}")

    # Convert to PIL Images
    frames_out = [Image.fromarray(arr).convert("RGB") for arr in frames_tensor]

    # Build timestamps (MM:SS) for each selected frame index
    timestamps = []
    for (start, end) in clip_times:
        timestamps.append(f"{int(start // 60):02d}:{int(start % 60):02d} to {int(end // 60):02d}:{int(end % 60):02d}")

    return frames_out, timestamps, duration_seconds

# Video Loader from sampled videos
##############################################################################
def load_image_directory_as_frames(
    folder_path: str,
    source_fps: float = 1.0,
    target_fps: float = 1.0,
    max_frames: int = 50,
    frames_per_clip: int = 1,
) -> Tuple[List[Image.Image], List[str]]:
    """
    Treats a directory of images as if they were consecutive frames in a 
    pseudo-video recorded at `source_fps`, then samples frames to achieve 
    an approximate `target_fps`, subject to a limit of `max_frames`.

    Args:
        folder_path (str): Directory path containing image frames (like "frame_001.jpg").
        source_fps (float): The framerate at which these images were presumably captured.
        target_fps (float): The approximate sampling rate we want in the output.
        max_frames (int): Hard limit on how many frames we return.

    Returns:
        (frames, timestamps):
          frames: List of loaded PIL.Image (RGB),
          timestamps: Parallel list of "MM:SS" strings indicating each frame's approximate time.

    Raises:
        RuntimeError: If `folder_path` doesn't exist or has no valid images,
                      or if we fail to load any frames after sampling.
    """
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"Path '{folder_path}' is not a directory.")

    # 1) Gather potential image files
    image_extensions = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    if not files:
        raise RuntimeError(f"No image files found in directory '{folder_path}'.")

    # 2) Extract numeric index from filenames, sort by (base, index)
    pattern = re.compile(r"(.*?)[-_]?(\d+)\..*$")
    numbered_files = []
    for fname in files:
        match = pattern.match(fname)
        if match:
            base, num_str = match.groups()
            try:
                num = int(num_str)
                numbered_files.append((fname, base, num))
            except ValueError:
                pass  # skip weird filenames
    if not numbered_files:
        raise RuntimeError(f"No valid numbered filenames found in '{folder_path}'.")

    # Sort primarily by base name, then by the numeric portion
    numbered_files.sort(key=lambda x: (x[1], x[2]))
    sorted_files = [nf[0] for nf in numbered_files]

    total_frames = len(sorted_files)
    # If no frames => we raise an error
    if total_frames == 0:
        raise RuntimeError(f"Directory '{folder_path}' appears empty after sorting valid images.")

    # 3) Compute the pseudo-video’s duration => total_frames / source_fps
    #    Then how many frames we want for target_fps => target_frames
    duration_seconds = total_frames / float(source_fps or 1.0)  # avoid dividing by 0
    estimated_frames = int(round(target_fps * duration_seconds)) if target_fps > 0 else max_frames

    # 4) Generate the final list of indices 
    frame_indices, clip_times = sample_clip_indices(
        frames_per_clip=frames_per_clip,           # or from model_args
        video_duration=duration_seconds,
        sampling_fps=target_fps,
        video_fps=source_fps,
        max_clips=max_frames
    )
    
    # If after removing duplicates we have nothing => fallback to single frame?
    if not frame_indices:
        frame_indices = [0]  # at least one
        
    # 5) Load frames
    frames = []
    for idx in frame_indices:
        img_path = os.path.join(folder_path, sorted_files[idx])
        try:
            img = Image.open(img_path).convert("RGB")
            frames.append(img)
        except Exception as e:
            logger.error(f"Failed to load image '{img_path}': {e}")
            # We skip the broken image
            continue
    
    timestamps = []
    for (start, end) in clip_times:
        timestamps.append(f"{int(start // 60):02d}:{int(start % 60):02d} to {int(end // 60):02d}:{int(end % 60):02d}")
    
    # If we ended up with zero loaded => raise
    if not frames:
        raise RuntimeError(f"No frames successfully loaded from '{folder_path}' after sampling.")
        
    return frames, timestamps, duration_seconds


# Image Loader
##############################################################################
def load_single_image(img_path: str) -> Image.Image:
    jpeg = Image.open(img_path)
    img = jpeg.copy().convert('RGB')
    return img


##############################################################################
# Helper Functions for Masking
##############################################################################
def find_global_img_patterns(tokens: List[str]) -> List[int]:
    mask_positions = []
    for i in range(len(tokens) - 4):
        if (
            tokens[i] == '<'
            and tokens[i+1] == 'global'
            and tokens[i+2] == '-'
            and tokens[i+3] == 'img'
            and tokens[i+4] == '>'
        ):
            mask_positions.extend([i, i+1, i+2, i+3, i+4])
    return mask_positions


def find_row_col_patterns(tokens: List[str]) -> List[int]:
    pattern = re.compile(r'^< row _ [1-9] _ col _ [1-9] >$')
    mask_positions = []
    for i in range(len(tokens) - 8):
        # Slice out exactly 9 tokens (e.g. <, row, _, 1, _, col, _, 1, >)
        group = tokens[i : i + 9]
        if pattern.fullmatch(" ".join(group)):
            mask_positions.extend(range(i, i + 9))
    return mask_positions


def _search_subsequence(
    sequence: torch.Tensor,
    pattern: List[int],
    start: int = 0
) -> int:
    """
    Searches for the first occurrence of 'pattern' in 'sequence'
    starting at offset 'start'. Returns the index of that occurrence,
    or -1 if not found.
    """
    # Convert input_ids to a Python list
    seq_list = sequence.tolist()
    pat_len = len(pattern)
    if pat_len == 0:
        return -1

    # Simple forward search
    for i in range(start, len(seq_list) - pat_len + 1):
        if seq_list[i : i + pat_len] == pattern:
            return i
    return -1


def _mask_system_tokens(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer
):
    """
    Identifies every occurrence of "System:" in `input_ids` (tokenized form),
    then masks (sets to IGNORE_INDEX) from the first token of "System:" up to 
    the next "<end_of_utterance>" marker or the end of the entire sequence.

    Args:
        input_ids (torch.Tensor): The token IDs for the conversation.
        labels (torch.Tensor): A copy of `input_ids` that we modify in-place 
           to set certain spans to IGNORE_INDEX.
        tokenizer: The tokenizer.
    """ 
    system_str = "System:"
    end_str    = "<end_of_utterance>"

    system_ids = tokenizer.encode(system_str, add_special_tokens=False)
    end_ids    = tokenizer.encode(end_str,   add_special_tokens=False)

    start_pos = 0
    while True:
        # 1) find next "System:"
        sys_start = _search_subsequence(input_ids, system_ids, start=start_pos)
        if sys_start == -1:
            break  # no more occurrences

        # 2) find next "<end_of_utterance>" after that
        sys_end = _search_subsequence(input_ids, end_ids, start=sys_start + len(system_ids))
        if sys_end == -1:
            sys_end = len(input_ids)  # if not found, go to end of sequence

        # 3) Mask [sys_start .. sys_end) in 'labels'
        labels[sys_start:sys_end] = IGNORE_INDEX

        # 4) Move forward
        start_pos = sys_end + len(end_ids)


def _mask_user_tokens(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer
):
    """
    Identifies every occurrence of "User:" in `input_ids`,
    then masks (sets to IGNORE_INDEX) from that token to the next "<end_of_utterance>" 
    or the end of the sequence. This removes the user's text from the training labels,
    so the model won't try to predict user text.

    Args:
        input_ids (torch.Tensor): The token IDs for the conversation.
        labels (torch.Tensor): A copy of `input_ids` that we modify in-place 
           to set certain spans to IGNORE_INDEX.
        tokenizer: The tokenizer.
    """
    user_str = "User:"
    end_str  = "<end_of_utterance>"

    user_ids = tokenizer.encode(user_str, add_special_tokens=False)
    end_ids  = tokenizer.encode(end_str,  add_special_tokens=False)

    start_pos = 0
    while True:
        # 1) find next "User:"
        usr_start = _search_subsequence(input_ids, user_ids, start=start_pos)
        if usr_start == -1:
            break  # no more occurrences

        # 2) find next "<end_of_utterance>" after that
        usr_end = _search_subsequence(input_ids, end_ids, start=usr_start + len(user_ids))
        if usr_end == -1:
            usr_end = len(input_ids)

        # 3) Mask [usr_start .. usr_end) in 'labels'
        labels[usr_start:usr_end] = IGNORE_INDEX

        # 4) Move forward
        start_pos = usr_end + len(end_ids)
        
##############################################################################
# Dataset
##############################################################################
class SupervisedDataset(Dataset):
    def __init__(
        self,
        dataset_args: Dict[str, Any],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        training_args: TrainingArguments,
        model_args: ModelArguments,
    ):
        """
        A dataset class that loads text/images/multi-image/videos, 
        tokenizes them via `processor`, and optionally masks user/system text.

        Args:
            dataset_args (Dict[str, Any]): Info specifying the dataset path, 
              sampling_strategy, possibly "source_fps", etc.
            processor (ProcessorMixin): Usually a multi-modal HF processor 
              that has a tokenizer + image_processor for vision.
            data_args (DataArguments): Contains config like `mask_user_tokens`, 
              `mask_system_tokens`, `fps`, etc.
            training_args (TrainingArguments): Possibly used for sampling or logging.
        """
        super().__init__()
        self.mask_user_tokens =  getattr(data_args, "mask_user_tokens", False)
        self.mask_system_tokens = getattr(data_args, "mask_system_tokens", True)
        self.add_media_intro_outro = getattr(data_args, "add_media_intro_outro", False)
        
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.training_args = training_args
        
        #todo: verfiery that args get here
        self.target_fps = getattr(model_args, "fps", 1.0) # CLIP sampling FPS
        self.frames_per_clip = int(getattr(model_args, "frames_per_clip", 1)) # NUMBER of frames/CLIP (to be averaged later)
        self.max_frames = getattr(data_args, "max_frames", 25)
        self.video_target_size = getattr(data_args, "video_target_size", 384)
        self.image_target_size = getattr(data_args, "image_target_size", 1536)
        self.data_folder = getattr(data_args, "data_folder", "")

        subdir = dataset_args.get("path", "")
        self.mm_path = os.path.join(self.data_folder, subdir)

        self.name = dataset_args.get("name", "unnamed_dataset")
        self.modality = dataset_args.get("modality", "unknown")
        self.source_fps = dataset_args.get("source_fps", 1)

        data_path = dataset_args["json_path"]
        self.list_data_dict = self._load_data(data_path)

        sampling_strategy = dataset_args.get("sampling_strategy", "all")
        self._apply_sampling_strategy(sampling_strategy)

        logger.info(
            f"[SupervisedDataset: {self.name}] - Label Masking Logic. "
            f"\nmask_user_tokens: {self.mask_user_tokens}, mask_system_tokens: {self.mask_system_tokens}\n"
        )
        logger.info(
            f"[SupervisedDataset: {self.name}] Final dataset size: {len(self.list_data_dict)}\n"
            f"Dataset Arguments - FPS: {self.target_fps}, "
            f"Max Frames: {self.max_frames}, "
            f"Video Target Size: {self.video_target_size}, "
            f"Image Target Size: {self.image_target_size}"
        )

    def _load_data(self, json_path: str) -> List[Dict[str, Any]]:
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        if json_path.endswith(".json"):
            with open(json_path, "r") as f:
                data = json.load(f)
        elif json_path.endswith(".jsonl"):
            data = []
            with open(json_path, "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported file format: {json_path}")

        logger.info(f"[{self.name}] Loaded {len(data)} items from {json_path}")
        return data

    def _apply_sampling_strategy(self, strategy: str):
        if strategy == "all":
            return
        if ":" not in strategy:
            return

        kind, amount_str = strategy.split(":")
        total = len(self.list_data_dict)

        if amount_str.endswith("%"):
            pct = float(amount_str.strip("%"))
            sampling_number = max(1, math.ceil(total * pct / 100.0))
        else:
            sampling_number = int(amount_str)

        if kind == "first":
            self.list_data_dict = self.list_data_dict[:sampling_number]
        elif kind == "end":
            self.list_data_dict = self.list_data_dict[-sampling_number:]
        elif kind == "random":
            random.seed(42)
            random.shuffle(self.list_data_dict)
            self.list_data_dict = self.list_data_dict[:sampling_number]

        logger.info(f"[{self.name}] after subsampling '{strategy}': {len(self.list_data_dict)} remain.")
    
    def __len__(self) -> int:
        return len(self.list_data_dict)
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                #next_index = min(i + 1, len(self.list_data_dict) - 1)
                random.seed(42) # TODO: should we set this here, or is this global variable we set anyway? make sure this makes sense. 
                next_index = random.choice(range(len(self.list_data_dict)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
            
    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[idx]
        if isinstance(idx, int):
            sources = [sources]
            
        content_type = sources[0].get("type", self.modality).lower()

        frames: List[Image.Image] = []
        timestamps: List[str] = []
        duration_seconds = None
        
        if content_type == "video":
            ## load videos
            #self.processor.image_processor.size = (self.video_target_size, self.video_target_size)
            self.processor.image_processor.size = {"longest_edge": self.video_target_size}
            self.processor.image_processor.do_resize = True
            self.processor.image_processor.do_image_splitting = False
            media = sources[0].get("video") or sources[0].get("image")
            if media:
                path = os.path.join(self.mm_path, media)
                if os.path.isdir(path):
                    ## TODO: can we simplify this logic??
                    frames, timestamps, duration_seconds = load_image_directory_as_frames(
                        folder_path=path,
                        source_fps=self.source_fps,
                        target_fps=self.target_fps,
                        max_frames=self.max_frames,
                        frames_per_clip=self.frames_per_clip
                    )
                else:
                    frames, timestamps, duration_seconds = load_video(
                        path,
                        max_frames=self.max_frames,
                        target_fps=self.target_fps,
                        frames_per_clip=self.frames_per_clip
                    )
                    
        elif content_type == "image" or content_type == "multiimage":

            ## load images and multi-image
            self.processor.image_processor.size = {"longest_edge": self.image_target_size}
            self.processor.image_processor.do_resize = True
            self.processor.image_processor.do_image_splitting = True
            media = sources[0].get("image", False)
            if media:
                if isinstance(media, str):
                    media = [media]
                paths = [os.path.join(self.mm_path, m) for m in media]
                frames = [load_single_image(path) for path in paths]
            else:
                raise("No image found for sample")
        else:
            frames = None
        
        conversations = copy.deepcopy([e["conversations"] for e in sources])
        ## get system message
        system_message = DEFAULT_SYSTEM_MESSAGE
        for k, v in sources[0].items():
            if isinstance(k, str) and "system" in k.lower() and "message" in k.lower() and isinstance(v, str):
                system_message = v
                break
                
        # Ensure each conversation has a system turn at index 0
        for conv in conversations:
            system_idx = next((i for i, t in enumerate(conv) if t.get("from", "").lower() == "system"), None)
            if system_idx is not None:
                # Move existing system turn to index 0
                conv.insert(0, conv.pop(system_idx))
            else:
                # If no system turn, add one
                conv.insert(0, {"from": "system", "value": system_message})
        
        conversations = [[self._convert_llava_to_openai_format(turn) for turn in conversation] for conversation in conversations]
        conversations = [self._replace_multimodal_tokens(conversation, content_type, frames, timestamps) for conversation in conversations]

        if self.add_media_intro_outro:
            for conv in conversations:
                if content_type == "text":
                    continue
                elif content_type == "image" or content_type == "multiimage":
                    if conv[1]['content'][0]['type'] == "image":
                        conv[1]['content'].insert(0, {'type': 'text', 'text': DEFAULT_IMAGE_INTRO})
                elif content_type == "video":
                    if conv[1]['content'][0]['type'] == "image" or conv[1]['content'][0]['type'] == "text" and FRAME_TIMESTAMP_MESSAGE_BASE in conv[1]['content'][0]['text']:
                        conv[1]['content'].insert(0, {'type': 'text', 'text': DEFAULT_VIDEO_INTRO.format(clip_count=num2words(len(frames)/self.frames_per_clip), video_duration=str(datetime.timedelta(seconds=duration_seconds)))})
                
                target_message_index = -1
                last_image_index = -1
                for i, message in enumerate(conv):
                    if 'content' in message:
                        for j, content in enumerate(message['content']):
                            if content.get('type') == 'image':
                                target_message_index = i
                                last_image_index = j
                
                # If we found an image, insert the outro right after it in the content list
                if target_message_index != -1 and last_image_index != -1:
                    conv[target_message_index]['content'].insert(last_image_index + 1,
                        {'type': 'text', 'text': DEFAULT_MEDIA_OUTTRO})

        text_input = self.processor.apply_chat_template(conversations[0], add_generation_prompt=False)
        encoded = self.processor(
            text=text_input,
            images=frames,
            return_tensors="pt",
            padding=False,
            allow_mismatch=True,
        )

        if encoded["input_ids"][0].size(0) > self.processor.tokenizer.model_max_length:
            raise ValueError(f"Sequence length {encoded['input_ids'][0].size(0)} exceeds maximum {self.processor.tokenizer.model_max_length}")
        
        # Each item is shape [1, seq_len]
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        # Start all labels as input_ids
        labels = input_ids.clone()
        self._mask_special_tokens(input_ids, labels)

        if self.mask_system_tokens:
            _mask_system_tokens(input_ids, labels, self.tokenizer)
            
        if self.mask_user_tokens:
            _mask_user_tokens(input_ids, labels, self.tokenizer)
            
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in encoded:
            out["pixel_values"] = encoded["pixel_values"][0]
        return out


    def _convert_llava_to_openai_format(self, llava_entry: Dict[str, str]) -> Dict[str, Any]:
        role_map = {"human": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
        speaker = llava_entry.get("from", "human").lower()
        role = role_map.get(speaker, "user")
    
        #text_value = llava_entry["value"].replace(DATA_VIDEO_TOKEN, DATA_IMAGE_TOKEN)
        text_value = llava_entry["value"]
        
        # Only replace <video> with <image> in user messages
        if role == "user":
            text_value = text_value.replace(DATA_VIDEO_TOKEN, DATA_IMAGE_TOKEN)
        else:
            # For assistant messages, replace tags with descriptive text
            text_value = text_value.replace(DATA_VIDEO_TOKEN, "video tag")
            text_value = text_value.replace(DATA_IMAGE_TOKEN, "image tag")
            
        # This regex splits on (<image>) but keeps it in the returned list
        parts = re.split(fr'({DATA_IMAGE_TOKEN})', text_value)
    
        content_list = []
        for chunk in parts:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk == "<image>":
                content_list.append({"type": "image"})
            else:
                content_list.append({"type": "text", "text": chunk})
    
        # Fallback if the text was empty or something went wrong
        if not content_list:
            content_list = [{"type": "text", "text": text_value}]
    
        return {"role": role, "content": content_list}
    
    def _replace_multimodal_tokens(
        self,
        conversation: List[Dict[str, Any]],
        content_type: str,
        frames: List[Image.Image],
        timestamps: List[Optional[str]],
    ) -> List[Dict[str, Any]]:
        """
        Post-processes a conversation to handle missing or expanded "image"/"video" tokens 
        based on the loaded frames. If there's no explicit placeholder but frames exist, 
        it injects them at the start of the user's first message. If there's exactly one
        placeholder token, it replicates it for every frame. This ensures the user's text 
        references the correct # of frames.
        #TODO: add logger.warning if no <image> placeholder!
    
        Args:
            conversation (List[Dict[str, Any]]): The conversation with "role" ("user"/"assistant") 
              and "content" (list of dict: {"type":..., "text":...}).
            content_type (str): "image" or "video". If "video" and multiple frames exist, 
              we replicate placeholders for each frame.
            frames (List[Image.Image]): The frames we loaded from the local path. 
              Could be 0 or more.
            timestamps (List[Optional[str]]): Timestamps (like "00:05") for each frame. 
              If some are None, we fallback to `i / self.target_fps`.
    
        Returns:
            conversation: The same conversation structure, but updated content blocks 
              so that each frame is referenced if needed. 
        """
    
        frames_inserted = False
        first_user_msg_processed = False
        
        total_frames = len(frames) if frames is not None else 0
        num_clips    = len(timestamps) if timestamps is not None else 0
        
        if num_clips * self.frames_per_clip != total_frames and content_type == 'video':
            logger.warning(
                f"Mismatch in {content_type}: {num_clips} clips * {self.frames_per_clip} != {total_frames} frames"
            )
        for msg in conversation:
            # We only modify the user's messages
            if msg["role"] != "user":
                continue
    
            # Check if there's an explicit image/video token in the user's content
            has_image_token = any(
                block["type"] in ("image", "video") 
                for block in msg["content"]
            )
            
            if not first_user_msg_processed and not has_image_token and frames:
                if content_type == "image":
                    # For a single-image scenario, just insert 1 placeholder
                    msg["content"].insert(0, {"type": "image"})
                    frames_inserted = True  # We won't re-insert later
                elif content_type == "video":
                    # Possibly multiple frames to insert. each clip needs ONE token !
                    new_blocks = []
                    for clip_idx in range(num_clips):
                        ts_str = timestamps[clip_idx]
                        new_blocks.append({"type": "text", "text": FRAME_TIMESTAMP_MESSAGE.format(clip_ts=ts_str)})
                        new_blocks.append({"type": "image"})
                    # Prepend to the user content
                    msg["content"] = new_blocks + msg["content"]
                    frames_inserted = True
    
            # Now we check placeholders inside the existing content to see if we want to expand them.
            updated_content = []
            for block in msg["content"]:
                if content_type == "video" and (block["type"] in ("image", "video")):
                    if not frames_inserted and frames:
                        for clip_idx in range(num_clips):
                            ts_str = timestamps[clip_idx]
                            # Possibly keep the user’s original block, if you like:
                            # updated_content.append(block)  
                            updated_content.append({
                                "type": "text",
                                "text": FRAME_TIMESTAMP_MESSAGE.format(clip_ts=ts_str),
                            })
                            updated_content.append({"type": "image"})
                        frames_inserted = True
                        # break out so we don't replicate placeholders multiple times
                        # if there are multiple <image>/<video> tokens in the user text
                        continue
                    else:
                        # Keep the user’s original block if you prefer:
                        updated_content.append(block)
                else:
                    # no changes
                    updated_content.append(block)

            msg["content"] = updated_content
            first_user_msg_processed = True

        return conversation

    def _mask_special_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor):
        labels[input_ids == self.tokenizer.pad_token_id] = IGNORE_INDEX
        
        if DEFAULT_IMAGE_TOKEN in self.tokenizer.additional_special_tokens:
            image_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            labels[input_ids == image_id] = IGNORE_INDEX

        if DEFAULT_VIDEO_TOKEN in self.tokenizer.additional_special_tokens:
            image_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_TOKEN)
            labels[input_ids == image_id] = IGNORE_INDEX

        if '<global-img>' in self.tokenizer.get_vocab():
            global_img_id = self.tokenizer.convert_tokens_to_ids('<global-img>')
            labels[input_ids == global_img_id] = IGNORE_INDEX
        
        image_patches = re.compile(r'<row_\d+_col_\d+>')
        patch_tokens = [token for token in self.tokenizer.get_vocab() if image_patches.fullmatch(token)]
        if len(patch_tokens) > 0:
            row_token_ids = self.tokenizer.convert_tokens_to_ids(patch_tokens)
            for token_id in row_token_ids:
                labels[input_ids == token_id] = IGNORE_INDEX
                
        # Possibly also ignore custom placeholders
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        positions_to_mask = find_global_img_patterns(tokens) + find_row_col_patterns(tokens)
        if len(positions_to_mask) > 0:
            logger.warn(f"found {len} global image + row col tokens not tokenized correctly!")
            
        for pos in positions_to_mask:
            labels[pos] = IGNORE_INDEX
            