from dotenv import load_dotenv
import os

from langsmith import traceable
load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


@traceable
def main():
    print("Hello from langchain-course!")
    information = """Квантова фізика — розділ теоретичної фізики, в якому вивчаються квантово-механічні та квантово-польові системи і закони їх руху. Основні закони квантової фізики вивчаються в рамках квантової механіки і квантової теорії поля і застосовуються в інших розділах фізики. Всі сучасні космологічні теорії також спираються на квантову механіку, яка описує поведінку атомних і субатомних частинок."""

    # summary_template = """Зроби список всіх слів в {information}. Слова в списку мають бути унікальними та стояти у своїй початковій формі."""
    summary_template = """Зроби короткий підсумок даного тексту: {information}. Підсумок має бути не більше 1 речення."""

    summary_prompt_template = PromptTemplate(
        input_vars=["information"],
        template=summary_template
    )

    # llm = ChatGoogleGenerativeAI( model="gemini-2.5-flash", temperature=0.2)
    llm = ChatOllama( model="gpt-oss:20b", temperature=0.2)

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print("Response:", response.content)
    

if __name__ == "__main__":
    main()
