from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import streamlit as st
import torch
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from ultralytics import YOLO
from tools import ActivityDetectionTool, VideoActivityRecognitionTool, VideoArmTrackingTool
from loguru import logger
from ultralytics import YOLO
from dotenv import load_dotenv
from  moviepy import VideoFileClip

from tools import ActivityDetectionTool, VideoActivityRecognitionTool, llm

load_dotenv()

YOLO_CLASSIFICATION_PATH = os.getenv('YOLO_CLASSIFICATION_PATH')
YOLO_TRACKING_PATH = Path("runs/detect/custom_yolo3/weights/best.pt") # TODO put in .env

class App:
    def __init__(self, device) -> None:
        self.llm = llm
        memory = MemorySaver()

        if "video_agent" not in st.session_state:
            video_tools = [VideoActivityRecognitionTool().setup()]
            st.session_state.video_agent = create_react_agent(llm, video_tools, checkpointer=memory)

        if "obj_agent" not in st.session_state:
            obj_tools = [ActivityDetectionTool().setup(YOLO(Path(YOLO_CLASSIFICATION_PATH).absolute(), verbose=False))]
            st.session_state.obj_agent = create_react_agent(llm, obj_tools, checkpointer=memory)

        if "arm_tracking" not in st.session_state:
            st.session_state.arm_tracking = VideoArmTrackingTool().setup(YOLO(Path(YOLO_TRACKING_PATH).absolute(), verbose=False))

    def process_video(self, clip_path):
        clip = VideoFileClip(clip_path)
        clip.write_videofile(clip_path.with_suffix(".mkv"))
        return clip_path.with_suffix(".mkv")
    
    def _process_file(self, prompt, file_path: str) -> None:
        try:
            config = {"configurable": {"thread_id": "abc123"}}
            
            # Extract necessary objects from session state to pass to worker functions
            video_agent = st.session_state.video_agent
            obj_agent = st.session_state.obj_agent
            arm_tracking = st.session_state.arm_tracking

            with ThreadPoolExecutor() as executor:
                # Submit the tasks to the executor
                future_video_description = executor.submit(get_video_description, video_agent, prompt, file_path, config) # NO
                future_object_response = executor.submit(get_object_response, obj_agent, prompt, file_path, config) # NO
                future_tracked_video = executor.submit(track_video, arm_tracking, file_path) # YES

                # Wait for all futures to complete and collect results
                for future in as_completed([future_video_description, future_object_response, future_tracked_video]):
                    if future == future_video_description:
                        video_description = future.result()
                    elif future == future_object_response:
                        object_response = future.result()
                    elif future == future_tracked_video:
                        tracked_video_directory = future.result()

            # After all tasks have completed, process the video to convert it
            tracked_video_directory = self.process_video(tracked_video_directory)

            # Log results
            logger.debug(f"{video_description=}")
            logger.debug(f"{object_response=}")
            logger.debug(f"{tracked_video_directory=}")
            return video_description["messages"][-1].content, object_response["messages"][-1].content, tracked_video_directory

        except Exception as e:
            logger.error(e)


    def _copy_file(self, uploaded_image):
        tmp_dir = "tmp/"
        os.makedirs(tmp_dir, exist_ok=True)
        temp_file_path = os.path.join(tmp_dir, uploaded_image.name)
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_image.getvalue())
    
    def _upload_image(self) -> None:
        uploaded_image = st.file_uploader("Upload an image")
        if uploaded_image:
            tmp_dir = "tmp/"
            os.makedirs(tmp_dir, exist_ok=True)
            temp_file_path = os.path.join(tmp_dir, uploaded_image.name)
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_image.getvalue())
            st.sidebar.image(temp_file_path, width=200)
            self._process_image(temp_file_path)

    def run(self) -> None:
        st.title("Chat with LLava")

        if "chat" not in st.session_state:
            st.session_state.chat = self.llm

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.form("chat_form"):
            prompt = st.text_input("Enter your message:")
            uploaded_file = st.file_uploader(
                "Upload an image (optional)",
                type=["jpg", "jpeg", "png", "mp4", "mkv", "avi"],
            )
            submit_button = st.form_submit_button("Send")

        # display the chat messages
        for message in st.session_state.messages:
            with st.chat_message(message.type):
                st.markdown(message.content.split(". Answer given")[0])
                if "file_path" not in message.additional_kwargs:
                    continue

                # image
                if message.additional_kwargs["file_path"].suffix[1:] in ["jpg", "jpeg", "png"]:
                    st.image(str(message.additional_kwargs["file_path"]))

                # video
                if message.additional_kwargs["file_path"].suffix[1:] in ["mp4", "mkv"]:
                    st.video(str(message.additional_kwargs["file_path"]))

        if submit_button:
            message = HumanMessage(content=prompt)

            if uploaded_file is not None:
                self._copy_file(uploaded_file)
                tmp_path = Path(f"tmp/{uploaded_file.name}")

                video_description, obj_detection, tracked_video_directory = self._process_file(message.content, tmp_path)
                message.content += ". Answer given these informations about the file, writing it in a more readable way for a user, don't write the file path:\n"
                if video_description:
                    message.content += f"{video_description=}\n"
                if obj_detection:
                    message.content += f"{obj_detection=}\n"

                # .additional_kwargs.update({"video_description":video, "object_description":obj})
                message.additional_kwargs.update({"file_path": tmp_path})

            # Append user's message to session state
            st.session_state.messages.append(message)
            with st.chat_message("user"):
                if prompt:
                    st.markdown(prompt)
                if uploaded_file is not None:
                    if "image" in uploaded_file.type:
                        st.image(uploaded_file)
                    if "video" in uploaded_file.type:
                        st.video(str(tracked_video_directory))

            # Get response from the LLM
            with st.chat_message("assistant"):
                print(st.session_state.messages)
                response = st.session_state.chat.invoke(st.session_state.messages)
                response.pretty_print()
                st.markdown(response.content)
                st.session_state.messages.append(response)
            print(f"{st.session_state.messages=}")




def get_video_description(agent, prompt, file_path, config):
    return agent.invoke(
        {"messages": [HumanMessage(content=f"{prompt}. Given the following video:{file_path}.")]}, config)

def get_object_response(agent, prompt, file_path, config):
    return agent.invoke(
        {"messages": [HumanMessage(content=f"{prompt}. Given the following video:{file_path}. Write it in a more readable way.")]}, config)

def track_video(agent, file_path):
    return agent.track(file_path)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()

