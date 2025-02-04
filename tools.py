import os
from pathlib import Path
import sys
import time
from typing import Optional
import cv2
from ultralytics import YOLO

from langchain.schema.runnable import RunnableLambda
from langchain.tools import BaseTool
from loguru import logger
from PIL import Image
import av
import numpy as np
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    BitsAndBytesConfig,
)
import torch

from utils import MAX_LENGTH, read_video_decord

REPO_ID = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    REPO_ID,
    quantization_config=quantization_config,
    device_map=device,
).to(device)

processor = LlavaNextVideoProcessor.from_pretrained(REPO_ID)


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def process_video(path, selected_frames=8):
    container = av.open(path)

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # # sample uniformly frames from the video
    # total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / selected_frames).astype(int)
    clip = read_video_pyav(container, indices)
    return clip


def run_llava(conversation):
    videos = []
    conversation_copy = map_conversation_to_llava_format(conversation, videos)
    prompt = processor.apply_chat_template(
        conversation_copy, add_generation_prompt=True
    )
    if videos:
        inputs_video = processor(
            text=prompt, videos=videos, padding=True, return_tensors="pt"
        ).to(model.device)
    else:
        inputs_video = processor(text=prompt, padding=True, return_tensors="pt").to(
            model.device
        )

    output = model.generate(**inputs_video, do_sample=False, max_new_tokens=50_000)
    generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
    _, _, generated_text = generated_text.rpartition("ASSISTANT:")
    return generated_text

def map_conversation_to_llava_format(conversation, videos):
    conversation_copy = []
    for conv in conversation:
        conv_line = {"role": conv["role"], "content": []}
        for item in conv["content"]:
            item_copy = item.copy()
            if item["type"] == "video":
                path = item_copy.pop("path")
                clip = process_video(path)
                videos.append(clip)
                # TODO add images

            conv_line["content"].append(item_copy)
        conversation_copy.append(conv_line)
    return conversation_copy


# Wrap it as a Runnable
llm = RunnableLambda(run_llava)


logger.remove()
logger.add(sys.stderr, level="INFO")


class VideoActivityRecognitionTool(BaseTool):
    name: str = "Video activity detection tool"
    description: str = "Classifies video content to detect and recognize activities performed by robotic tools, enabling accurate action identification for automated analysis."

    llm: Optional[LlavaNextVideoForConditionalGeneration] = None

    def setup(
        self, llm: LlavaNextVideoForConditionalGeneration
    ) -> "VideoActivityRecognitionTool":
        self.llm = llm
        return self

    def _run(self, video_path: str) -> str:
        logger.debug(f"Video path: {video_path}")
        video = read_video_decord(video_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are working in an industrial setting where robotic arms perform various activities. Your task is to analyze videos of these robotic arms in action and accurately classify the specific activity being performed in each video. Answer only with the activity detected.",
                    },
                    {"type": "video"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        batch = processor(
            text=prompt,
            videos=video,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        video = video.to(llm.device)

        out = llm.generate(
            **batch, pixel_values_videos=video, max_length=MAX_LENGTH, do_sample=True
        )
        generated_text = processor.batch_decode(out, skip_special_tokens=True)

        logger.debug(f"Generated prompt: {generated_text}")
        return generated_text

    def _arun(self, query: str):
        raise NotImplementedError


class ObjectDetectionTool(BaseTool):
    name: str = "Object detection on image tool"
    description: str = (
        "Perform object detection on an image (read an image path) with a text prompt."
    )

    model: Optional[YOLO] = None

    def setup(self, _model: YOLO) -> "ObjectDetectionTool":
        self.model = _model
        return self

    def _run(self, image_path, prompt: str) -> str:
        logger.debug(f"Image path: {image_path}, prompt: {prompt}")

        if not self.model:
            raise ValueError(
                "The model has not been set up. Please call `setup` first."
            )

        result = self.model(image_path["title"])
        output_dir = "output/"
        os.makedirs(output_dir, exist_ok=True)
        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_img_path = os.path.join(output_dir, f"{now_time}_obj_detection.png")
        result[0].save(filename=output_img_path)

        return Image.open(output_img_path)

    def _arun(self, query: str):
        raise NotImplementedError
