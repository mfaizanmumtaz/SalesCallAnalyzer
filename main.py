from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import ConfigurableField
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import shelve
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"]="Audo Evaluation"

FEW_SHORT_TRAINING_ONE = ""
TERMS_AND_CONDITIONS = ""
TRANSCRIPTION = ""

with shelve.open("few_short_prompt") as db:
    db = db["few_short_prompt"]
    TERMS_AND_CONDITIONS = db.get("tac")
    TRANSCRIPTION = db.get("transcription")
    FEW_SHORT_TRAINING_ONE = db.get("few_short_training_one")

model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4).configurable_alternatives(
    ConfigurableField(id="llm"),
    openai=ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0.4),
    default_key="google"
).with_fallbacks([ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4)])

content_moderation_chain = ChatPromptTemplate.from_messages([
    ("user",TRANSCRIPTION),
    ("assistant",FEW_SHORT_TRAINING_ONE),
    ("user","""You are a content moderation assistant.You will be provided company terms and conditions, along with a company employee phone call transcription. You must follow these steps to complete the task:

> Analyze the entire terms and conditions and the phone call transcription.
> Check whether the employee is following company terms and conditions or not.
> If the employee is not following company terms and conditions, take those actual parts of the transcription where employee violate for each violation part add a prefix that says "Violation 1: ..."
"Violation 2: ...".
     
> Generate a concise report for each violation part how employee is doing Violation of company Terms and Conditions using markdown syntax.

> If the employee does not make any violations, respond with just 'No violations identified. 
  The employee followed the company terms and conditions during the phone call transcription.'

> Please do your best it is very important to my career.
     
-----------

> <Terms and conditions>: {tac} <Terms and conditions>

-----------

> <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")]) | model | StrOutputParser()

objection_prompt = ChatPromptTemplate.from_messages(
    [
        ("user","""You are objection detector assistant in phone calls.Plase follow these steps to complete the task:
        > Did customer make any objection in this following phone call?.
        > Please do your best it is very important to my career.
         
        ----------------
         
        > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")
    ]
) | model | StrOutputParser()

sales_techniques = ChatPromptTemplate.from_messages(
    [
        ("user", """You are sales advisor techniques analyzer.Please follow these steps to complete the task:
        > How good is sales advisor techniques in this following phone call.
        > Please do your best it is very important to my career.
         
        ----------------
         
        > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")
    ]
) | model | StrOutputParser()

summarizer_prompt = ChatPromptTemplate.from_messages(
    [
    ("user","""You are phone calls summarizer assistant.Please follow these steps to complete the task:
    > Generate the Summarize of this following phone call.
    > Please do your best it is very important to my career.

    ----------------
     
    > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")

]) | model | StrOutputParser()

sentiment_chain = ChatPromptTemplate.from_messages([
    ("user","""You are phone calls sentiment analyzer.Please follow these steps to complete the task:
    > Determine the sentiment of this following phone call.
    > Please do your best it is very important to my career.
     
    ----------------
     
    > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")

]) | model | StrOutputParser()

map_chain = RunnableParallel(Customer_Objections=objection_prompt,Employee_Sales_Techniques=sales_techniques,Summary_Of_Call=summarizer_prompt,Overall_Call_Sentiment=sentiment_chain)

import streamlit as st
import assemblyai as aai

st.set_page_config("Phone Call Analyzer", layout="wide")

st.title("Sales Advisor Phone Call Analyzer")
st.write("This app will analyze a sales advisor phone call and provide you with a report on the call. The report will include information on the call's content, sentiment, and sales techniques.")

def transcriber():
    with st.sidebar:
        st.title("Phone Call Analyzer")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3","m4a",'ogg'])
        if audio_file is not None:
            if "audio" not in os.listdir():
                os.mkdir("audio")
            audio_file_path = os.path.join("audio",audio_file.name)
            with st.spinner("Audio file is being transcript."):
                try:
                    with open(audio_file_path, "wb") as f:
                        f.write(audio_file.read())

                    config = aai.TranscriptionConfig(
                        speaker_labels=True)
                    
                    transcriber = aai.Transcriber()
                    transcript = transcriber.transcribe(audio_file_path, config).utterances 
                    return "".join([f"Speaker {utterance.speaker}: {utterance.text}\n\n" for utterance in transcript])

                except Exception as e:
                    st.warning("Error",e)
                finally:
                    os.remove(audio_file_path)          

transcript = transcriber()
if transcript:
    with st.spinner("Please wait..."):
        response=map_chain.with_config(configurable={"llm":"google"}).invoke({"tac": TERMS_AND_CONDITIONS,"transcription": transcript})
        for key,value in response.items():
            st.subheader(key.upper())
            st.write(value)
# i have add a new code here