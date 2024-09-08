import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough

import json
from dotenv import load_dotenv


class OutputParser(BaseOutputParser):
    def parse(self, output: str):
        return output.replace("(o)", "").strip()


function = {
    "name": "generate_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

with st.sidebar:
    user_api_key = st.text_input("API Key")
if not user_api_key:
    st.warning("API Key가 필요합니다.")
    st.stop()

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    api_key=user_api_key,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
).bind(
    function_call={
        "name": "generate_quiz",
    },
    functions=[
        function,
    ],
)


def format_docs(docs):
    docs = docs["context"]
    return "\n\n".join([document.page_content for document in docs])


question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    and user can choose difficulty level. if user choose 'easy', the question will be easy such as 'What is the color of the ocean?' and if user choose 'hard', the question will be hard and specific such as 'Which is not correct answer about Pacific Ocean?'.
    True means the answer is correct and False means the answer is incorrect. and remove True and False from the final result.

    Question examples:

    Question: What is the color of the ocean?
    Answers: 
    Red (False)
    Yellow (False)
    Green (False)
    Blue (True)

    Question: What is the capital or Georgia?
    Answers: 
    Baku (False)
    Tbilisi(True)
    Manila(False)
    Beirut(False)

    Question: When was Avatar released?
    Answers: 
    2007(False)
    2001(False)
    2009(True)
    1998(False)

    Question: Who was Julius Caesar?
    Answers: 
    A Roman Emperor (True)
    Painter (False)
    Actor (False)
    Model (False)

    Your turn!
    difficulty level: {difficulty_level}
    Context: {context}
""",
        )
    ]
)


@st.cache_data(show_spinner="파일을 읽는 중입니다...")
def split_file(file):
    file_content = file.read()
    file_path = f"./c.cache/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path=file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="퀴즈를 생성하는 중입니다...")
def generate_quiz_chain(_docs, topic):
    chain = (
        {
            "context": format_docs,
            "difficulty_level": RunnablePassthrough(),
        }
        | question_prompt
        | llm
    )
    result = chain.invoke({"context": _docs, "difficulty_level": difficulty_level})
    return result


@st.cache_data(show_spinner="퀴즈를 생성하는 중입니다...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=1)
    docs = retriever.get_relevant_documents(topic)
    return docs


with st.sidebar:
    docs = None
    if user_api_key != "":
        choice = st.selectbox(
            "Choose a document",
            ["Wikipedia", "Upload a file"],
        )
        difficulty_level = st.selectbox(
            "난이도를 선택하세요.",
            ["easy", "hard"],
        )
        if choice == "Upload a file":
            file = st.file_uploader(
                """파일을 올려주세요""", type=["txt", "docx", "pdf"], help=".txt .docx .pdf 파일만 업로드 가능합니다."
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("검색할 주제를 입력하세요")
            if topic:
                docs = wiki_search(topic)

    st.link_button(
        "Github",
        url="https://github.com/2wonbin/nomad-quiz-gpt",
        help="해당 프로젝트의 깃허브 레포지토리로 이동합니다.",
        use_container_width=True,
    )

if not docs:
    st.markdown(
        """
        ## 📚 Quiz GPT를 찾아주셔서 감사합니다!
        
        ### Quiz를 생성하고, 생성된 Quiz를 풀어보세요!

        ### 좌측 사이드바에서 Wikipedia나 파일을 업로드하면 Quiz GPT를 사용할 수 있습니다.
        """
    )
else:
    response = generate_quiz_chain(docs, topic)
    questions = response.additional_kwargs["function_call"]["arguments"]
    parsed_questions = json.loads(questions)
    perfect_score = len(parsed_questions["questions"])
    score = 0
    with st.form("questions_form"):
        for question in parsed_questions["questions"]:
            st.write(question["question"])
            value = st.radio("정답에 체크하세요.", [answer["answer"] for answer in question["answers"]], index=None)
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("정답입니다!")
                score += 1
            elif value is not None:
                st.error("틀렸습니다.")

        button = st.form_submit_button(label="제출하기")
        if button == True:
            if perfect_score == score:
                st.balloons()
                st.success(f"모든 문제를 맞추셨습니다! 점수: {score}/{perfect_score}")
            else:
                st.error(f"틀린 문제가 있습니다. 확인 후 다시 제출해주세요")
