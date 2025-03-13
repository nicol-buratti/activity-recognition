# Activity recognition LLM tool
Agentic LLM specialized for activity recognition tasks of robotic arms in a healthcare environment.

This program is designed as a chat interface, enabling users to perform classification tasks on videos, track robotic arms, and perform video reasoning.

## Features:
- **Activity Classification**: Classifies actions and activities performed by robotic arms.
- **Robotic Arm Tracking**: Tracks movements and positions of robotic arms within video frames.
- **Video Reasoning**: Performs reasoning over video data for advanced task understanding.

## Installation
To install the necessary dependencies, run the following commands:

### 1. Clone the Repository  
```bash
git clone https://github.com/nicol-buratti/activity-recognition.git
cd activity-recognition
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```
### 3. Activate the Virtual Environment
  - On Windows
    ```bash
    venv\Scripts\activate
    ```
  - On macOS/Linux
    ```bash
    source venv/bin/activate
    ```
### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Additional Dependencies
- **Ollama**: Required for LLM processing and chat capabilities.
- **PyTorch with CUDA**: Required for deep learning and video processing. Ensure you have the correct CUDA version for your system.

## Environment Setup
Before running the program, you must create a .env file with the necessary environment variables. You can copy the example from .envexample:

```bash
cp .envexample .env
```
Fill in the required information in the .env file, such as model name or other configuration details.

## Running the Application
To run the program, use Streamlit to start the front-end interface:

```bash
streamlit run ./frontend.py
```
This will launch the chat interface where you can interact with the system and perform activity recognition tasks on videos.

# Contributors
- [Buratti Nicol](https://github.com/nicol-buratti)
- [Pennesi Diego](https://github.com/Diezz01)
- [Reucci Filippo](https://github.com/reus702)
