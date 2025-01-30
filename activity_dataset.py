from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import  random_split

ACTION_CLIPS_PATH = "atlas_dione_objectdetection\ATLAS_Dione_ObjectDetection\ATLAS_Dione_ObjectDetection_Study_ActionClips\ATLAS_Dione_ObjectDetection_Study_ActionClips"


class VideoDataset(Dataset):
    def __init__(self, directory_label_dict):
        """
        Initialize the dataset.

        Args:
            directory_label_dict (dict): A dictionary where keys are directory paths (Path objects)
                                         and values are labels.
        """
        self.data = []

        for dir_path, label in directory_label_dict.items():
            if not dir_path.is_dir():
                continue
            for file_path in dir_path.iterdir():
                if file_path.is_file() and self._is_video_file(file_path):
                    self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
    def _is_video_file(self, file_path):
      """Check if the file is a video based on its extension, and ignore files starting with '._'."""
      if file_path.name.startswith('._'):
          return False

      # Check if the file has a valid video extension
      video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
      return file_path.suffix.lower() in video_extensions


def get_dataset_splits(train_size = 0.8):
    dataset = VideoDataset(mapping)

    test_size = 1.0 - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

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
