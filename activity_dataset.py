from pathlib import Path
import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, BitsAndBytesConfig
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, directory_label_dict, transform=None):
        """
        Initialize the dataset.

        Args:
            directory_label_dict (dict): A dictionary where keys are directory paths (Path objects)
                                         and values are labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.transform = transform

        for dir_path, label in directory_label_dict.items():
            if not dir_path.is_dir():
                continue
            for file_path in dir_path.iterdir():
                if file_path.is_file() and self._is_video_file(file_path):
                    self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        label = lab_num[label]
        frames = self._load_video(video_path)
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        return self.collate_fn(frames, label)

    def _is_video_file(self, file_path):
      """Check if the file is a video based on its extension, and ignore files starting with '._'."""
      # Ignore files starting with '._'
      if file_path.name.startswith('._'):
          return False

      # Check if the file has a valid video extension
      video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
      return file_path.suffix.lower() in video_extensions


    def _load_video(self, video_path):#original one
        """Load a video file and sample frames using torchvision.io.read_video."""
        video_path = str(video_path)  # Convert to string as required by read_video
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit = "sec", output_format="TCHW") # Compose transformation are compatible with this output format

        return frames
    
    def collate_fn(self, video, label):
        # Let's use chat template to format the prompt correctly
        print(video.shape)
        conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Provide a detailed caption for this video."},
                        {"type": "video"},
                        ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": num_lab[label]},
                        ],
                },
            ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

        batch = processor(
            text=prompt,
            videos=video,
            truncation=True,
            # max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return batch
    
    ##PADDING
    def _load_video(self, video_path, target_length=28):
        """Carica un video e applica il padding o il campionamento."""
        video_path = str(video_path)
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")
        num_frames = frames.size(0)

        # # Campionamento o padding
        # if num_frames > max_frames:
        #     indices = torch.linspace(0, num_frames - 1, steps=max_frames).long()
        #     frames = frames[indices]
        # elif num_frames < max_frames:
        #     pad_frames = max_frames - num_frames
        #     padding = torch.zeros((pad_frames, frames.size(1), frames.size(2), frames.size(3)), dtype=frames.dtype)
        #     frames = torch.cat([frames, padding], dim=0)

        return frames
        # C, T, H, W = frames.shape

        # if T != target_length:
        #     # Use interpolation to resample the video frames
        #     video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        #     video_tensor = F.interpolate(video_tensor, size=(target_length, H, W), mode='trilinear', align_corners=False)
        #     video_tensor = video_tensor.squeeze(0)  # Remove batch dimension

        # return video_tensor



def get_dataset_splits(train_size = 0.8):
    dataset = VideoDataset(mapping, transformation)

    test_size = 1.0 - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset



ACTION_CLIPS_PATH = "atlas_dione_objectdetection\ATLAS_Dione_ObjectDetection\ATLAS_Dione_ObjectDetection_Study_ActionClips\ATLAS_Dione_ObjectDetection_Study_ActionClips"
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MAX_LENGTH = 11000

processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)

action_clips_path = Path(ACTION_CLIPS_PATH)
directories = [item for item in action_clips_path.iterdir() if item.is_dir() and not item.name == "set12"]

mapping = {"set00":"1 arm placing",
            "set01":"2 arms placing",
            "set02":"Placing Rings",
            "set03":"Placing Rings 2 arms",
            "set04":"Pull Off",
            "set05":"Pull Through",
            "set06":"Suture Pick Up",
            "set07":"UVA Pick Up",
            "set08":"Suture Pull Through",
            "set09":"UVA Pull Through",
            "set10":"Suture Tie",
            "set11":"UVA Tie",
             }
mapping = {dir: mapping[dir.name] for dir in directories}

lab_num = {item: i for i, item in enumerate(mapping.values())}
num_lab = {item : i for i, item in lab_num.items()}
# define video tranformations, maybe they could be skipped
from activity_dataset import VideoDataset


transformation = transforms.Compose([
    # PadOrTruncateFrames(20),
    transforms.Resize((336, 336)),
    # transforms.Resize((70, 70)),  # Resize to LLava next video dimension
    transforms.GaussianBlur(kernel_size=3),  # Denoising
    # # transforms.RandomRotation(degrees=10),  # Augmentation
    transforms.ConvertImageDtype(torch.float),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # For RGB images
])

dataset = VideoDataset(mapping, transformation)