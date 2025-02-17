from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import streamlit as st
import torch
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentType
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor
from ultralytics import YOLO
from langchain import hub
from tools import ObjectDetectionTool, VideoActivityRecognitionTool
from langchain.memory import ConversationBufferMemory
from loguru import logger
from ultralytics import YOLO

from tools import ObjectDetectionTool, VideoActivityRecognitionTool, llm

YOLO_PATH = "runs/classify/custom_yolo7/weights/best.pt" # TODO add env variable

class App:
    def __init__(self, device) -> None:
        self.llm = llm
        memory = MemorySaver()

        if "video_agent" not in st.session_state:
            video_tools = [VideoActivityRecognitionTool().setup()]
            st.session_state.video_agent = create_react_agent(llm, video_tools, checkpointer=memory)

        if "obj_agent" not in st.session_state:
            obj_tools = [ObjectDetectionTool().setup(YOLO(Path(YOLO_PATH).absolute(),verbose = False))]
            st.session_state.obj_agent = create_react_agent(llm, obj_tools,  checkpointer=memory)

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

    def _process_file(self, prompt, file_path: str) -> None:
        try:
            config = {"configurable": {"thread_id": "abc123"}}
            with ProcessPoolExecutor() as ex: #TODO parallel execution
                video_description = st.session_state.video_agent.invoke({"messages":[HumanMessage(content=prompt + f". Given the following video:{file_path}.")]}, config)
                object_response = st.session_state.obj_agent.invoke({"messages":[HumanMessage(content=prompt + f". Given the following video:{file_path}.")]}, config)
            
            logger.debug(f"{video_description=}")
            logger.debug(f"{object_response=}")

            return video_description["messages"][-1].content, object_response["messages"][-1].content
        except Exception as e:
            logger.error(e)


    def _copy_file(self, uploaded_image):
        tmp_dir = "tmp/"
        os.makedirs(tmp_dir, exist_ok=True)
        temp_file_path = os.path.join(tmp_dir, uploaded_image.name)
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_image.getvalue())

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
                type=["jpg", "jpeg", "png", "mp4", "mkv"],
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
                
                video_description, obj_detection = self._process_file(message.content, tmp_path)
                message.content += ". Answer given these informations about the file, don't write the file path:\n"
                if video_description:
                    message.content += f"{video_description=}\n"
                if obj_detection:
                    message.content += f"{obj_detection=}\n"
                
                # .additional_kwargs.update({"video_description":video, "object_description":obj})
                message.additional_kwargs.update({"file_path":tmp_path})

            # Append user's message to session state
            st.session_state.messages.append(message)
            with st.chat_message("user"):
                if prompt:
                    st.markdown(prompt)
                if uploaded_file is not None:
                    if "image" in uploaded_file.type:
                        st.image(uploaded_file)
                    if "video" in uploaded_file.type:
                        st.video(uploaded_file)

            # Get response from the LLM
            with st.chat_message("assistant"):
                print(st.session_state.messages)
                response = st.session_state.chat.invoke(st.session_state.messages)
                response.pretty_print()
                st.markdown(response.content)
                st.session_state.messages.append(response)
            print(f"{st.session_state.messages=}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()
