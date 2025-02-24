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
from langchain_core.prompts.chat import ChatPromptTemplate

load_dotenv()

YOLO_CLASSIFICATION_PATH = os.getenv('YOLO_CLASSIFICATION_PATH')
YOLO_TRACKING_PATH = Path("runs/detect/custom_yolo3/weights/best.pt") # TODO put in .env

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}"),
    ]
)

def modify_messages(messages: list):
    return chat_prompt_template.invoke({"messages": messages})

class App:
    def __init__(self, device) -> None:

        if "chat" not in st.session_state:
            st.session_state.chat = llm

        if "agent" not in st.session_state:
            memory = MemorySaver()

            video_tools = [
                VideoActivityRecognitionTool().setup(), 
                ActivityDetectionTool().setup(YOLO(Path(YOLO_CLASSIFICATION_PATH).absolute(), verbose=False))
                ]

            st.session_state.agent = create_react_agent(llm, video_tools, checkpointer = memory, messages_modifier = modify_messages)

        if "tracking" not in st.session_state:
            st.session_state.tracking = VideoArmTrackingTool().setup(YOLO(Path(YOLO_TRACKING_PATH).absolute(), verbose=False))

    def process_video_to_mkv(self, clip_path):
        clip = VideoFileClip(clip_path)
        clip.write_videofile(clip_path.with_suffix(".mkv"))
        return clip_path.with_suffix(".mkv")

    def _copy_file(self, uploaded_image):
        tmp_dir = "tmp/"
        os.makedirs(tmp_dir, exist_ok=True)
        temp_file_path = Path(f"{tmp_dir}/{uploaded_image.name}")
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_image.getvalue())
        return temp_file_path
    
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
        st.title("Chat with RoboArmVision")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.form("chat_form"):
            prompt = st.text_input("Enter your message:")
            uploaded_file = st.file_uploader(
                "Upload an image (optional)",
                type=["mp4", "mkv", "avi"],
            )
            submit_button = st.form_submit_button("Send")

        # display the chat messages
        for message in st.session_state.messages:
            with st.chat_message(message.type):
                st.markdown(message.content)
                if "file_path" not in message.additional_kwargs:
                    continue

                # image TODO
                # if message.additional_kwargs["file_path"].suffix[1:] in ["jpg", "jpeg", "png"]:
                #     st.image(str(message.additional_kwargs["file_path"]))

                # video
                if message.additional_kwargs["file_path"].suffix in [".mp4", ".mkv"]:
                    st.video(str(message.additional_kwargs["file_path"]))

        if submit_button:
            message = HumanMessage(content=prompt)

            if uploaded_file is not None:
                tmp_path = self._copy_file(uploaded_file)
                tracker = st.session_state.tracking

                with ProcessPoolExecutor() as executor:
                # Submit the tasks to the executor
                    tracked_video_directory = executor.submit(tracker.track, tmp_path)
                
                message.additional_kwargs.update({"file_path": tmp_path})

            # Append user's message to session state
            st.session_state.messages.append(message)
            with st.chat_message(message.type):
                if prompt:
                    st.markdown(prompt)
                if uploaded_file is not None:
                    if "image" in uploaded_file.type:
                        st.image(uploaded_file)
                    if "video" in uploaded_file.type:
                        st.video(str(tmp_path))

            # Get response from the LLM
            with st.chat_message("assistant"):
                config = {"configurable": {"thread_id": "1"}}
                if uploaded_file:
                    mess = st.session_state.messages.copy()
                    mess[-1].content += f" {tmp_path}" # TODO inject path not via prompt
                    response = st.session_state.agent.invoke({"messages": mess}, config)
                    response = response["messages"][-1]
                    st.markdown(response.content)
                    video = self.process_video_to_mkv(tracked_video_directory.result())
                    st.video(str(video))
                    response.additional_kwargs["file_path"] = video
                else:
                    response = st.session_state.chat.invoke(st.session_state.messages)
                    st.markdown(response.content)

                # response.pretty_print()
                # logger.debug(f"{response=}")
                st.session_state.messages.append(response)
            print(f"{st.session_state.messages=}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()
