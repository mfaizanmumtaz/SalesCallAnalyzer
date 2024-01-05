from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import ConfigurableField
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"]="Audo Evaluation"

model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0).configurable_alternatives(
    ConfigurableField(id="llm"),
    openai=ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0),
    default_key="google"
).with_fallbacks([ChatGoogleGenerativeAI(model="gemini-pro",temperature=0)])

# Getting data from utils.py
from utils import get_data
GetData = get_data()
Quality_Assurance_Scorecard_Questions = GetData.get_questions
transcript = GetData.get_transcription
VULNERABILITY_EXAMPLES = GetData.get_vulnerability_examples

prompt = ChatPromptTemplate.from_messages([

    ("user","""You are a helpful assistant who analyzes the advisor Mortgage & Equity Release Call.Advisor phone call transciption with customer is delimited in XML tag.Please follow these steps to complete the task.\
    step 1 - Analyze adisor phone call transcription with the customer perfectly and understand it before moving to any conclusion or decision.\
    step 2 - Answer the following question.
    Question - {question}

    Use the following output format:
    Separate your answers with Python \\n line breaks that can be used in streamlit app to parse it.
    Question: <User Question>
    Answer: <Answer in Yes Or No Depending on your analysis>
    Explanation: <One sentence consice explanation>
    
    Note: Please do not generate any extra text and always do your best to follow the above-written instructions. It is very important for my career.

    -------------

    > <advisor phone call transcription with customer>: {transcription} <advisor phone call transcription with customer>""")])

chain = prompt | model | StrOutputParser()

Quality_Assurance_Scorecard_Chain = RunnablePassthrough() | (lambda x: [{"question":question["question"],"transcription":x["transcription"] } for question in Quality_Assurance_Scorecard_Questions]) | chain.map() | (lambda x: "\n\n".join(x).replace("\n", "<br>"))

vulnerability_prompt = ChatPromptTemplate.from_messages([

    ("user","""You are a vulnerability detector in customer support calls in customer speech.please follow these below steps to complete the task.\
        > Vulnerabilities examples in CSV formate are delimited with ``` and customer support call is delimited with XML tag.\
        > If you could find these kinds of vulnerabilities in customer speech, then create a concise report where an customer behave unexpectedliy like For example in the Uk if a customer says to an advisor “I am seriously ill”, the advisor should show care for the customer , ask about their illness and let that information help them decide if they should continue or stop the call
        That is described as customer vulnerability.\
        > Please do your best it is very important to my career.\
        
        ---------------
        
        > ```vulnerability examples```: {vulnerability_examples} ```vulnerability examples```
         
        ---------------
         
        > <Employee phone call transcription with customer>: {transcription} </Employee phone call transcription with customer>
         """)
])

vulnerability_prompt_chain = RunnablePassthrough.assign(
    vulnerability_examples= lambda x: VULNERABILITY_EXAMPLES
) | vulnerability_prompt | model | StrOutputParser()

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

map_chain = RunnableParallel(Quality_Assurance_Scorecard=Quality_Assurance_Scorecard_Chain,Vulnerability=vulnerability_prompt_chain,Customer_Objections=objection_prompt,Employee_Sales_Techniques=sales_techniques,Summary_Of_Call=summarizer_prompt,Overall_Call_Sentiment=sentiment_chain)

import streamlit as st
import assemblyai as aai

st.set_page_config("Phone Call Analyzer", layout="wide",page_icon="icons/ico.png")

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
        res=map_chain.with_config(configurable={"llm":"google"}).invoke({"transcription": transcript})      
        for index,(title,description) in enumerate(res.items(),start=1):
            if title == "Quality_Assurance_Scorecard":
                st.markdown(f"#### {index}. Quality Assurance For Mortgage And Equity Release")
            elif title == "Vulnerability":
                st.markdown(f"#### {index}. {title}")
            elif title == "Customer_Objections":
                st.markdown(f"#### {index}. Customer Objections")
            elif title == "Customer_Objections":
                st.markdown(f"#### {index}. Customer Objections")
            elif title == "Employee_Sales_Techniques":
                st.markdown(f"#### {index}. Employee Sales Techniques")
            elif title == "Summary_Of_Call":
                st.markdown(f"#### {index}. Summary Of Call")
            elif title == "Overall_Call_Sentiment":
                st.markdown(f"#### {index}. Overall Call Sentiment")
            st.markdown(description,unsafe_allow_html=True)
