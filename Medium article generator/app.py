import os
from apikey import apikey

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


os.environ["OPENAI_API_KEY"] = apikey

st.title('Medium Article generator')
topic = st.text_input('Input your topic of interest')
# language = st.text_input('Input Language')

# topic_update = f'{topic} give a long article'

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Give me a medium article title on {topic}'
)

article_template = PromptTemplate(
    input_variables = ['title'],
    template = 'Give me a medium article for title: {title}'
)

llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

llm2 = ChatOpenAI(model_name='gpt-4-turbo', temperature=0.9)

article_chain = LLMChain(llm=llm2, prompt=article_template, verbose=True)

overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)

if topic:
    # response = llm(title_template.format(topic=topic,language='english'))
    response = overall_chain.run(topic)
    st.write(response)