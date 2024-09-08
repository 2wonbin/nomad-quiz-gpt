import os
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnableLambda


class JsonOputputParser(BaseOutputParser):
    def parse(self, text: str):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOputputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    api_key=api_key,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)


def format_docs(docs):
    return "\n\n".join([document.page_content for document in docs])


question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    and user can choose difficulty_level, and your answer should be based on that. For example, if the user chooses 'easy', the questions should be gernerate quiz with elementary level of difficulty. if you choose 'hard', the questions should be gernerate quiz with advanced level of difficulty such as PH.D level.
    Use (o) to signal the correct answer.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    Your turn!

    difficulty_level: "hard"
    Context: {context}
""",
        )
    ]
)

difficulty_level_runnable = RunnableLambda(lambda x: {"difficulty_level": x})
question_chain = {"difficulty_level": difficulty_level_runnable, "context": format_docs} | question_prompt | llm
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
    
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
        
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
        
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
        
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
    Example Output:
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
    }}
    ```
    Your turn!

    Questions: {context}
    """,
        )
    ]
)

formatting_chain = formatting_prompt | llm


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
    chain = {"context": question_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="퀴즈를 생성하는 중입니다...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=1)
    return retriever.get_relevant_documents(topic)


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose a document",
        ["Wikipedia", "Upload a file"],
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

if not docs:
    st.markdown(
        """
        
        ## 📚 Quiz GPT를 찾아주셔서 감사합니다!
        
        ### Quiz를 생성하고, 생성된 Quiz를 풀어보세요!

        ### 좌측 사이드바에서 Wikipedia나 파일을 업로드하면 Quiz GPT를 사용할 수 있습니다.
        """
    )

else:
    response = generate_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio("정답에 체크하세요.", [answer["answer"] for answer in question["answers"]], index=None)
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("정답입니다!")
            elif value is not None:
                st.error("틀렸습니다.")

        button = st.form_submit_button()
