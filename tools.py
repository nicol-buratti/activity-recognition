from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time
from typing import Optional
import torchvision
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
import torch.nn.functional as F

import cv2
import torch
from langchain.chat_models import init_chat_model

from training_notebooks.llava.utils import MAX_LENGTH, read_video_decord
from dotenv import load_dotenv

from twfich import FINCH
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

        clusters, clusters_videos = VideoClustering.cluster(video_path)
        # with ThreadPoolExecutor() as ex:
            # results = ex.map(self.model, clusters)
        clusters_results = [self.model(c, stream=True) for c in clusters_videos]

        clusters_classes = []
        for results in clusters_results:
            cluster_class = Counter((res.names[res.probs.top1] for res in results)).most_common(n=1)[0]
            clusters_classes.append(cluster_class)
        
        logger.debug(f"{clusters_classes=}")
        logger.debug(f"{clusters=}")

        return clusters_classes

    def _arun(self, query: str):
        raise NotImplementedError


class VideoClustering:

    @staticmethod
    def get_partition(clusters, n):
        start = 0
        intervals = []
        for i, _ in enumerate(clusters):
            if clusters[i][n] != clusters[start][n]:
                intervals.append((start, i - 1))
                start = i
        intervals.append((start, len(clusters) - 1))
        return intervals
    
    @staticmethod
    def cluster(video_path):
        video, _ , metadata = torchvision.io.read_video(Path(video_path), pts_unit = "sec", output_format="TCHW")
        fps = metadata["video_fps"]

        video_vectors = video.reshape(video.size(0), -1)

        # Temporally-Weighted Hierarchical Clustering
        c, num_clust, _ = FINCH(video_vectors, tw_finch=True)

        p = len(num_clust) - 1
        logger.debug(f"clusters = {num_clust[-1]}")
        clusters = VideoClustering.get_partition(c, p)
        logger.debug(clusters)
        logger.debug(f"{video.shape=}")


        resized_video = F.interpolate(video, size=(640, 640), mode='bilinear', align_corners=False).float()
        logger.debug(f"{resized_video.dtype=}")

        clusters_videos = [resized_video[start:end] for start, end in clusters]

        return clusters, clusters_videos
