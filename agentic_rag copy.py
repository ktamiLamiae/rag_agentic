from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool, create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv


import os
load_dotenv(override=True)
print(os.getenv("OPENAI_API_KEY"))


embedding_model = OpenAIEmbeddings()

chunks = [
    "Je m'appelle Ahmed Benali, je suis ingénieur en informatique spécialisé en intelligence artificielle",
    "Je travaille à l'Université Mohammed V de Rabat en tant que chercheur",
    "J'ai obtenu mon master en 2018 puis mon doctorat en 2022",
    "Je suis passionné par la technologie, la science et l'innovation",
    "J'aime aussi lire des livres et écrire sur des sujets liés à l'IA",
    "Je suis originaire de Fès, une ville historique du Maroc",
    "Après le lycée, j'ai intégré une école d'ingénieurs où j'ai étudié pendant 5 ans",
    "J'ai participé à plusieurs projets en data science et machine learning",
    "En parallèle de mon travail, je continue à apprendre de nouvelles technologies",
    "Mon objectif est de contribuer au développement de solutions intelligentes pour la société"
]

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    collection_name="cv_information"
)

retriever = vector_store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="cv_tool",
    description="Get information about Ahmed Benali CV"
)

@tool
def get_employee_info(name: str):
    """
    Get information about a given employee (name, salary, seniority)
    """
    print("get_employee_info tool invoked")
    return {"name": name, "salary": 12000, "seniority": 5}


@tool
def send_email(email: str, subject: str, content: str):
    """
    Send email with subject and content
    """
    print(f"Sending email to {email}, subject : {subject}, content : {content}")
    return f"email successfully sent to {email}, subject : {subject}, content : {content}"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[send_email, get_employee_info, retriever_tool],
    system_prompt="Answer to user query using provided tools"
)

resp = agent.invoke(
    {"messages": [HumanMessage(content="Quel est le salaire de yassine ?")]}
)

print(resp["messages"][-1].content)