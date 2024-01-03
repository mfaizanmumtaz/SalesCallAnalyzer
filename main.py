from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import ConfigurableField
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import shelve
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["ASSEMBLYAI_API_KEY"] = "e68c5a0ba5a145debfb06a9ef3029dc2"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCoqBakUtcaayyOfU26fd-AODycWpeAp3M"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"]="Audo Evaluation"
os.environ["LANGCHAIN_API_KEY"] = "ls__0d2d5a023b1c4771b9ba042af91cb78a"

FEW_SHORT_TRAINING_ONE = ""
TERMS_AND_CONDITIONS = ""
VULNERABILITY_EXAMPLES = ""

with shelve.open("few_short_prompt") as db:
    db = db["few_short_prompt"]
    TERMS_AND_CONDITIONS = db.get("tac")
    FEW_SHORT_TRAINING_ONE = db.get("few_short_training_one")
    VULNERABILITY_EXAMPLES = db.get("vulnerability_examples")

model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4).configurable_alternatives(
    ConfigurableField(id="llm"),
    openai=ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0.4),
    default_key="google"
).with_fallbacks([ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4)])

Quality_Assurance_Scorecard_Question = [
    {"question": "Has the advisor completed the correct Data Protection Act (DPA) procedures, including obtaining and verifying customer consent for data processing?"},
    {"question": "Did the advisor explain how the customer's data will be used and stored?"},
    {"question": "Has the advisor not only made the customer aware of FCA regulation but also provided a brief explanation of its implications for the customer?"},
    {"question": "Did the advisor clarify potential risks associated with secured loans and ensure the customer understands?"},
    {"question": "Has the mortgage advisor introduced themselves, the company, and confirmed the purpose of the call at the beginning?"},
    {"question": "Has the advisor confirmed the customer's nationality and explained the relevance to the mortgage process?"},
    {"question": "Did the advisor confirm if the customer has received relevant literature on the company's services before the call?"},
    {"question": "Has the advisor consistently used open questioning techniques to encourage detailed responses from the customer?"},
    {"question": "Did the advisor employ a mix of open and closed questions to gather necessary information efficiently?"},
    {"question": "Has the advisor questioned the customer's basic eligibility for a secured loan, including probing about any potential challenges in meeting eligibility criteria?"},
    {"question": "Has the advisor explained the significance of the customer's credit history in the mortgage application process?"},
    {"question": "Did the advisor inquire about any existing financial commitments that might impact loan eligibility?"},
    {"question": "Has the advisor not only read but ensured comprehension and appropriate delivery of mandatory scripts, including credit file, affiliate, and fair processing notice?"},
    {"question": "Did the advisor provide additional context or clarification where necessary to enhance customer understanding of scripted information?"},
    {"question": "Has the advisor conducted a detailed affordability assessment, considering the customer's income, expenses, and financial stability?"},
    {"question": "Did the advisor explain how changes in interest rates could affect monthly repayments and overall affordability?"},
    {"question": "Has the advisor thoroughly explored and documented the customer's short-term and long-term goals with the loan, including potential changes in circumstances?"},
    {"question": "Did the advisor inquire about the customer's risk tolerance and preferences regarding loan terms?"},
    {"question": "Has the advisor effectively and diplomatically challenged any inconsistent or unclear customer responses, ensuring a comprehensive understanding?"},
    {"question": "Did the advisor provide examples or scenarios to help the customer clarify their preferences and requirements?"},
    {"question": "Has the advisor not only made a recommendation but also provided a clear rationale, ensuring alignment with the customer's stated goals and circumstances?"},
    {"question": "Did the advisor discuss potential risks associated with the recommended solution and explore alternatives?"},
    {"question": "What specific actions or communication strategies could the advisor have employed to enhance compliance, customer interaction, or recommendation quality?"},
    {"question": "Did the advisor actively seek feedback from the customer about their experience during the call?"},
    {"question": "Has the advisor demonstrated the ability to identify signs of customer vulnerability, such as financial difficulties, mental health issues, or other challenging circumstances?"},
    {"question": "Did the advisor handle the interaction with sensitivity, empathy, and follow established protocols for dealing with vulnerable customers?"},
    {"question": "Has the advisor questioned how the vulnerability may impact the customer in the short or long term?"},
    {"question": "Has the advisor questioned the history of the vulnerability, and if any treatment has been prescribed, and on what frequency?"},
    {"question": "Has the advisor gained consent to the customer's personal information, such as the vulnerability?"},
    {"question": "Has the advisor demonstrated a clear understanding and adherence to FCA regulations, MCOB (Mortgage Conduct of Business), and the Equity Release Council Handbook throughout the call?"},
    {"question": "Did the advisor ensure compliance with in-house policies related to customer interactions and mortgage processes?"},
    {"question": "Have there been any deviations from regulatory guidelines or company policies, and were they appropriately addressed by the advisor?"},
    {"question": "Has the advisor discussed and documented any agreed-upon follow-up actions or next steps with the customer?"},
    {"question": "Did the advisor explain the timeline for processing the customer's application and set realistic expectations?"},
    {"question": "How well has the advisor demonstrated understanding of the customer's unique situation and concerns throughout the call?"},
    {"question": "Did the advisor summarise key points and agreements reached during the call to ensure mutual understanding?"}
]

prompt = ChatPromptTemplate.from_messages([

    ("user","""You are a helpful assistant who analyzes the advisor Mortgage & Equity Release Call.Advisor phone call transciption with customer is delimited in XML tag.Please follow these steps to complete the task.\
    step 1 - Analyze adisor phone call transcription with the customer perfectly and understand it before moving to any conclusion or decision.\
    step 2 - Answer the following question.
    Question - {question}

    Use the following output format:
    Question: <User Question>
    Answer: <Answer in Yes Or No Depending on your analysis>
    Explanation: <One sentence consice explanation>
    
    Separate your answers with line breaks.
     
    Note: Please do not generate any extra text and always do your best to follow the above-written instructions. It is very important for my career.

    -------------

    > <advisor phone call transcription with customer>: {transcription} <advisor phone call transcription with customer>""")])

chain = prompt | model | StrOutputParser()

Quality_Assurance_Scorecard_Chain = RunnablePassthrough() | (lambda x: [{"question":question["question"],"transcription":x["transcription"] } for question in Quality_Assurance_Scorecard_Question[:12]]) | chain.map() | (lambda x: "\n\n".join(x))

# print(Quality_Assurance_Scorecard_Chain.invoke({"transcription":"I am looking good."}))

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

vulnerability_prompt_chain.invoke({"transcription":"we are best"})

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
        res=map_chain.with_config(configurable={"llm":"google"}).invoke({"transcription": transcript})      
        for title,description in res.items():
            st.subheader(title.upper())
            st.write(description)