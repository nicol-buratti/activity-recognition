import base64
import os
from ultralytics import YOLO
import streamlit as st
import torch
from langchain.agents import initialize_agent
from langchain.agents.agent import  AgentType
from langchain_core.messages import HumanMessage

from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from loguru import logger

from tools import ObjectDetectionTool, PromptGeneratorTool

llm = Ollama(model="llava:7b")

class App:
    def __init__(self, device) -> None:

        # if "agent" not in st.session_state:
        self._agent = initialize_agent(
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            tools=[
                ObjectDetectionTool().setup(
                    YOLO(
                        "C:/Users/Informatica_UNICAM/Desktop/activity-recognition/"
                        "runs/detect/custom_yolo7/weights/best.pt"
                    )
                ),
                PromptGeneratorTool().setup(llm),
            ],
            llm=llm,
            memory=ConversationBufferMemory(return_messages=True),
            verbose=True,
            max_iterations=3,
        )
        st.session_state["agent"] = self._agent

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

    def _process_image(self, image_path: str) -> None:
        try:
            result = self._agent(
                f"Describe the image: {image_path} and detect objects with the description."
            )
            logger.debug(result)
        except Exception as e:
            logger.error(e)

    def run(self) -> None:
        st.title("Chat with LLava")
        if "chat" not in st.session_state:
            client = llm
            st.session_state.chat = client

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.form("chat_form"):
            prompt = st.text_input("Enter your message:")
            uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
            submit_button = st.form_submit_button("Send")
            
        if submit_button:
            message_content = []
            if prompt:
                message_content.append({"type": "text", "text": prompt})
            if uploaded_file is not None:
                # Read the image data and encode it in base64
                image_bytes = uploaded_file.read()
                image_type = uploaded_file.type  # e.g., 'image/jpeg'
                image_data = base64.b64encode(image_bytes).decode("utf-8")
                # Include the image data in the message content
                print( {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_data}"}})
                message_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_data}"}}
                )

            message = HumanMessage(content=message_content)
            # Append user's message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                if prompt:
                    st.markdown(prompt)
                if uploaded_file is not None:
                    st.image(uploaded_file)
            # Get response from the LLM
            response = st.session_state.chat.invoke([message])
            # Append assistant's response to messages
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()
