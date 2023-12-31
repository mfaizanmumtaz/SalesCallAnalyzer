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
VULNERABILITY_EXAMPLES = ""

with shelve.open("few_short_prompt") as db:
    db = db["few_short_prompt"]
    TERMS_AND_CONDITIONS = db.get("tac")
    TRANSCRIPTION = db.get("transcription")
    FEW_SHORT_TRAINING_ONE = db.get("few_short_training_one")
    VULNERABILITY_EXAMPLES = db.get("vulnerability_examples")

model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4).configurable_alternatives(
    ConfigurableField(id="llm"),
    openai=ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0.4),
    default_key="google"
).with_fallbacks([ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4)])

content_moderation_chain = ChatPromptTemplate.from_messages([
    ("user",TRANSCRIPTION),
    ("assistant",FEW_SHORT_TRAINING_ONE),
    ("user","""You are a content moderation assistant in customer support call.You will be provided general terms and conditions (delimited with XML tags), along with customer support phone call (delimited with XML tags). You must follow these steps to complete the task:
> If the employee is not following company terms and conditions, take those actual parts of the transcription where employee violate for each violation part add a prefix that says "Violation 1: ..."
"Violation 2: ...".     
> Generate a concise report for each violation part how employee is doing wrong using markdown syntax.
> If the employee does not make any violations, respond with just 'No violations identified. 
  The employee followed the terms and conditions during the phone call.'
> Please do your best it is very important to my career.
     
-----------

> <Terms and conditions>: {tac} <Terms and conditions>

-----------

> <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")]) | model | StrOutputParser()

vulnerability_prompt = ChatPromptTemplate.from_messages([
    ("user","""You are a vulnerability detector in customer support calls in customer speech.please follow these below steps to complete the task.\
        > Vulnerabilities examples in CSV formate are delimited with ``` and customer support call is delimited with XML tag.\
        > If you could find these kinds of vulnerabilities in customer speech, then create a concise report where an customer behave unexpectedliy like For example in the Uk if a customer says to an advisor “I am seriously ill”, the advisor should show care for the customer , ask about their illness and let that information help them decide if they should continue or stop the call
        That is described as customer vulnerability.\
        > Please do your best it is very important to my career.\
        
        ---------------
        
        > ```vulnerability examples```: {vulnerability_examples} ```vulnerability examples```
         
        ---------------
         
        > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>
         """)
]) | model | StrOutputParser()

objection_prompt = ChatPromptTemplate.from_messages(
    [
        ("user","""You are an objection detector assistant in customer support call (delimited with XML tags).Plase follow these steps to complete the task:
        > Did customer make any objection in this following phone call?.
        > If a customer makes any objections, then mark down key points for each objection raised by the customer.        
        > Please do your best it is very important to my career.
         
        ----------------
         
        > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")
    ]
) | model | StrOutputParser()

sales_techniques = ChatPromptTemplate.from_messages(
    [
        ("user", """You are Customer Support Call employee techniques analyzer (delimited with XML tags).Please follow these steps to complete the task:
        > How good is Customer Support Call advisor techniques in this following phone call.
        > Please give consice report.
        > Please do your best it is very important to my career.
         
        ----------------
         
        > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")
    ]
) | model | StrOutputParser()

summarizer_prompt = ChatPromptTemplate.from_messages(
    [
    ("user","""Summarize the following customer support call.\
    which is delimited with XML tag in a single paragraph. Then write a markdown list of the speakers and each of their key points. Finally, list the next steps or action items suggested by the speakers, if any.
     
    -------------
     
    > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")

]) | model | StrOutputParser()

sentiment_chain = ChatPromptTemplate.from_messages([
    ("user","""You are customer support call sentiment analyzer (delimited with XML tags).Please follow these steps to complete the task:
    > Determine the sentiment of this following phone call.
    > Please do your best it is very important to my career.
     
    ----------------
     
    > <Employee phone call transcription with customer>: {transcription} <Employee phone call transcription with customer>""")

]) | model | StrOutputParser()

map_chain = RunnableParallel(Vulnerability=vulnerability_prompt,Customer_Objections=objection_prompt,Employee_Sales_Techniques=sales_techniques,Summary_Of_Call=summarizer_prompt,Overall_Call_Sentiment=sentiment_chain)

import streamlit as st
import assemblyai as aai

st.set_page_config("Phone Call Analyzer", layout="wide")

st.title("Customer Support Call Analyzer")
st.write("This app will analyze Customer Support Call and provide you with a report on the call. The report will include information on the call's content, sentiment, and sales techniques.")

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
        res=map_chain.with_config(configurable={"llm":"google"}).invoke({"transcription": transcript,"vulnerability_examples":VULNERABILITY_EXAMPLES})      
        for k,v in res.items():
            st.subheader(k.upper())
            st.write(v)