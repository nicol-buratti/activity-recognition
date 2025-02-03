import os
import sys
import time
from typing import Optional
from ultralytics import YOLO

from langchain.schema.runnable import RunnableLambda
from langchain.tools import BaseTool
from loguru import logger
from PIL import Image

from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch

from utils import MAX_LENGTH, read_video_decord

REPO_ID = "cams01/LLaVa-robot-activity-recognition"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    REPO_ID, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(REPO_ID)

# TODO add video
def run_llava(conversation, video = None):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    batch = processor(
        text=prompt,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)


    out = model.generate(**batch, max_length = MAX_LENGTH, do_sample = True)
    generated_text = processor.batch_decode(out, skip_special_tokens = True)
    _,_,generated_text = generated_text[0].partition("ASSISTANT: ")
    return generated_text

# Wrap it as a Runnable
llm = RunnableLambda(run_llava)


logger.remove()
logger.add(sys.stderr, level="INFO")

#llm = Ollama(model="llava:7b")

class VideoActivityRecognitionTool(BaseTool):
    name: str = "Video activity detection tool"
    description: str = "Classifies video content to detect and recognize activities performed by robotic tools, enabling accurate action identification for automated analysis."

    llm: Optional[LlavaNextVideoForConditionalGeneration] = None

    def setup(self, llm: LlavaNextVideoForConditionalGeneration) -> "VideoActivityRecognitionTool":
        self.llm = llm
        return self

    def _run(self, video_path: str) -> str:
        logger.debug(f"Video path: {video_path}")
        video = read_video_decord(video_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are working in an industrial setting where robotic arms perform various activities. Your task is to analyze videos of these robotic arms in action and accurately classify the specific activity being performed in each video. Answer only with the activity detected."},
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

        out = llm.generate(**batch, pixel_values_videos = video, max_length = MAX_LENGTH, do_sample = True)
        generated_text = processor.batch_decode(out, skip_special_tokens = True)

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
            raise ValueError("The model has not been set up. Please call `setup` first.")

        result = self.model(image_path["title"])
        output_dir = "output/"
        os.makedirs(output_dir, exist_ok=True)
        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_img_path = os.path.join(output_dir, f"{now_time}_obj_detection.png")
        result[0].save(filename = output_img_path)

        return Image.open(output_img_path)

    def _arun(self, query: str):
        raise NotImplementedError
