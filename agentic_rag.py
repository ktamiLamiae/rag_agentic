from langchain.agents import create_agent

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool

# import os
load_dotenv(override=True)
# print(os.getenv("OPENAI_API_KEY"))

chunks = [
    "Je m'appelle Ahmed Benali, je suis ingénieur en informatique spécialisé en intelligence artificielle",
    "Je travaille à l'Université Mohammed V de Rabat en tant que chercheur",
    "J'ai obtenu mon master en 2018 puis mon doctorat en 2022",
    "Je suis passionné par la technologie, la science et l'innovation",
    # "J'aime aussi lire des livres et écrire sur des sujets liés à l'IA",
    "Je suis originaire de Fès, une ville historique du Maroc",
    # "Après le lycée, j'ai intégré une école d'ingénieurs où j'ai étudié pendant 5 ans",
    # "J'ai participé à plusieurs projets en data science et machine learning",
    # "En parallèle de mon travail, je continue à apprendre de nouvelles technologies",
    # "Mon objectif est de contribuer au développement de solutions intelligentes pour la société"
]
embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    collection_name="cv_collection"
)

# vector_store.embedding_function = embedding_model

retriever = vector_store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="cv_tool",
    description="Get information about Ahmed Benali CV"
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def get_employee_info(name: str):
    """
    Get information about a given employee (name, salary, seniority)
    """
    # print("get_employee_info tool invoked")
    return {"name": name, "salary": 12000, "seniority": 5}


@tool
def send_email(email: str, subject: str, content: str):
    """
    Send email with subject and content
    """
    # print(f"Sending email to {email}, subject : {subject}, content : {content}")
    return f"email successfully sent to {email}, subject : {subject}, content : {content}"


agent = create_agent(
    model=llm,
    tools=[retriever_tool,send_email, get_employee_info],
    system_prompt="Answer to user query using provided tools"
)

# resp = agent.invoke(
#     {"messages": [HumanMessage(content="Quel est le salaire de yassine ?")]}
# )

# print(resp["messages"][-1].content)