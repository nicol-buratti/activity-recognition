import os
from pathlib import Path
import sys
import time
from typing import Optional
from ultralytics import YOLO

from langchain.schema.runnable import RunnableLambda
from langchain.tools import BaseTool
from loguru import logger
from PIL import Image
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    BitsAndBytesConfig,
)
import torch
from langchain.chat_models import init_chat_model

from training_notebooks.llava.utils import MAX_LENGTH, read_video_decord
from dotenv import load_dotenv
load_dotenv()

REPO_ID = os.getenv('REPO_ID')

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "llama3.2:1b" # TODO add to .env
llm = init_chat_model(model_id, model_provider="ollama", temperature=0.1)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

class VideoActivityRecognitionTool(BaseTool):
    name: str = "Video Activity Description Tool"
    description: str = (
        "Generates a descriptive summary of the activities or scenes in a video, only if provided. "
        "Provide the video path to receive an activity description based on its content."
    )

    llm: Optional[LlavaNextVideoForConditionalGeneration] = None
    processor:Optional[LlavaNextVideoProcessor] = None

    def setup(self) -> "VideoActivityRecognitionTool":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            REPO_ID,
            quantization_config=quantization_config,
            device_map=device,
        ).to(device)

        model_processor = LlavaNextVideoProcessor.from_pretrained(REPO_ID)
        self.llm = model
        self.processor = model_processor
        return self

    def _run(self, video_path: str) -> str:
        if not (Path(video_path).exists() and Path(video_path).is_file()):
            return "no video"

        logger.debug(f"Video path video recognition: {video_path}")
        video = read_video_decord(video_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this video",
                    },
                    {"type": "video"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs_video = self.processor(
                    text=prompt, videos=video, padding=True, return_tensors="pt"
                ).to(self.llm.device)

        output = self.llm.generate(**inputs_video, do_sample=False, max_new_tokens=50_000)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        _, _, generated_text = generated_text.rpartition("ASSISTANT: ")
        logger.debug(f"{generated_text=}")
        return generated_text

    def _arun(self, query: str):
        raise NotImplementedError


class ObjectDetectionTool(BaseTool):
    name: str = "Video Object Detection Tool"
    description: str = (
        "Performs object detection on a video file. Given the video path, it identifies and returns objects detected in the video frames."
    )

    model: Optional[YOLO] = None

    def setup(self, _model: YOLO) -> "ObjectDetectionTool":
        self.model = _model
        return self

    def _run(self, video_path) -> str:
        if not (Path(video_path).exists() and Path(video_path).is_file()):
            return "no video"
        logger.debug(f"Video path object detection: {video_path}")

        results =  self.model(str(Path(video_path).absolute()))
        return [res.names[res.probs.top1] for res in results]

    def _arun(self, query: str):
        raise NotImplementedError
