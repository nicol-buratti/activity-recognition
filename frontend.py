import os
import streamlit as st
import torch
from langchain.agents import initialize_agent
from langchain.agents.agent import  AgentType
from langchain_core.messages import HumanMessage

from langchain.memory import ConversationBufferMemory
from loguru import logger

from tools import VideoActivityRecognitionTool, llm

# llm = Ollama(model="llava:7b")

class App:
    def __init__(self, device) -> None:

        # if "agent" not in st.session_state:
        self._agent = initialize_agent(
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            tools=[
                # ObjectDetectionTool().setup(
                #     YOLO(
                #         "C:/Users/Informatica_UNICAM/Desktop/activity-recognition/"
                #         "runs/detect/custom_yolo7/weights/best.pt"
                #     )
                # ),
                VideoActivityRecognitionTool().setup(llm),
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
            
        # display the chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"][0]["text"]) # TODO integrate images/video

        if submit_button:
            content = []
            message = {"role": "user", "content": content}
            if prompt:
                content.append({"type": "text", "text": prompt})
            if uploaded_file is not None:
                if "image" in uploaded_file.type:
                    content.append({"type": "image", "path" : uploaded_file})
                if "video" in uploaded_file.type:
                    content.append({"type": "video"})

            # Append user's message to session state
            st.session_state.messages.append(message.copy()) #TODO test with files
            with st.chat_message("user"):
                if prompt:
                    st.markdown(prompt)
                if uploaded_file is not None:
                    st.image(uploaded_file)
            # Get response from the LLM
            response = st.session_state.chat.invoke([message])
            # Append assistant's response to messages
            st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text":response}]})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()
