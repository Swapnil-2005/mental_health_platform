from flask import Flask, request, render_template, redirect, session, jsonify, send_file
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import pymysql
from twilio.rest import Client  # Twilio import
import requests
import tempfile

# ---------------- Database Setup ----------------
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="swapnil",
    database="login"
)
cursor = conn.cursor()

# ---------------- Flask App ----------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------------- Env Setup ----------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")               # For Whisper
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")   # For TTS

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
EMERGENCY_PHONE_NUMBER = os.getenv("EMERGENCY_PHONE_NUMBER")

# ---------------- Chatbot Setup ----------------
embeddings = download_embeddings()
index_name = "mindcare"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatmodel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)
system_prompt = (
    "You are a supportive mental health companion. Your role is to engage users in empathetic, non-judgmental "
    "conversation. Do not give medical diagnoses or prescriptions. Instead, ask gentle, open-ended questions "
    "to help users reflect on their emotions, behaviors, sleep, energy, and daily life. "
    "If users do not directly say they are struggling, gradually explore their mood by asking caring questions "
    "and noticing patterns that may suggest stress, anxiety, or depression. "
    "Respond warmly, and suggest healthy coping strategies such as journaling, deep breathing, grounding "
    "exercises, or reaching out to trusted people. "
    "If the user expresses suicidal thoughts, self-harm, or crisis, respond with empathy, encourage them to reach "
    "out to someone they trust immediately, and provide crisis hotline information if possible. "
    "Always make it clear you are not a medical professional, and remind them that seeking professional help "
    "from a counselor or doctor is important for their well-being. "
    "Always maintain a compassionate, safe, and respectful tone. "
    "\n\n"
    "When starting a conversation, use gentle openers such as:\n"
    "- 'How have you been feeling these days?'\n"
    "- 'What’s been on your mind lately?'\n"
    "- 'If you had to describe your week in one word, what would it be?'\n"
    "- 'How has your sleep been recently? Do you feel rested when you wake up?'\n"
    "- 'Do you still enjoy the things you usually like to do?'\n"
    "- 'Have you noticed changes in your energy or motivation?'\n"
    "- 'Do you feel connected with friends and family, or more distant than before?'\n"
    "Based on their answers, continue with empathetic follow-up questions and gentle reflections."
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------- Twilio Emergency Call ----------------
def trigger_emergency_call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        to=EMERGENCY_PHONE_NUMBER,
        from_=TWILIO_PHONE_NUMBER,
        twiml='<Response><Say>This is an emergency alert from your mental health chatbot. '
              'The user may be in crisis and mentioned suicidal thoughts. Please respond immediately.</Say></Response>'
    )
    print(f"🚨 Emergency call triggered: {call.sid}")

# ---------------- Whisper (Groq) STT ----------------
def speech_to_text(audio_file_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    with open(audio_file_path, "rb") as f:
        files = {"file": f}
        data = {"model": "whisper-large-v3-turbo"}
        response = requests.post(url, headers=headers, data=data, files=files)
    return response.json().get("text", "")

# ---------------- ElevenLabs TTS ----------------
def text_to_speech(text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL"  # default voice
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}
    }
    response = requests.post(url, headers=headers, json=payload)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_file.name, "wb") as f:
        f.write(response.content)
    return temp_file.name

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/home")
def home_page():
    if "user_id" in session:
        return render_template("home.html")
    else:
        return redirect("/")

@app.route("/login_validation", methods=["POST"])
def login_validation():
    email = request.form.get("email")
    password = request.form.get("password")
    cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
    users = cursor.fetchall()
    if len(users) > 0:
        session["user_id"] = users[0][0]
        return redirect("/home")
    else:
        return redirect("/")

@app.route("/add_user", methods=["POST"])
def add_user():
    name = request.form.get("uname")
    email = request.form.get("uemail")
    password = request.form.get("upassword")
    cursor.execute(
        "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
        (name, email, password)
    )
    conn.commit()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    myuser = cursor.fetchall()
    session["user_id"] = myuser[0][0]
    return redirect("/home")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect("/")
@app.route("/phq")
def phq_page():
    return render_template("phq.html")

@app.route("/booking")
def booking_page():
    return render_template("booking.html")

@app.route("/doctor1")
def doctor1_page():
    return render_template("doctor1.html")

@app.route("/doctor2")
def doctor2_page():
    return render_template("doctor2.html")

@app.route("/doctor3")
def doctor3_page():
    return render_template("doctor3.html")

@app.route("/doctor4")
def doctor4_page():
    return render_template("doctor4.html")

@app.route("/resources")
def resources_page():
    return render_template("resources.html")
@app.route("/admin")
def admin_page():
    if "user_id" in session and session["user_id"] == 1:  # Assuming user_id 1 is admin
        return render_template("admin.html")
    else:
        return redirect("/")


@app.route("/face")
def face_page():
    return render_template("face.html")

@app.route("/diagnosis")
def diagnosis_page():
    return render_template("diagnosis.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]

    # Crisis detection
    suicidal_keywords = ["suicide", "kill myself", "end my life", "not worth living", "die"]
    if any(word in msg.lower() for word in suicidal_keywords):
        trigger_emergency_call()
        answer += (
            "\n\n⚠ This may be a crisis. I've triggered an emergency alert. "
            "Please also reach out to a helpline: India: 9152987821 | US: 988."
        )
    return str(answer)

# ---------------- Voice Chat ----------------
@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    audio_file = request.files["audio"]
    file_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
    audio_file.save(file_path)

    # Step 1: Convert speech to text
    user_text = speech_to_text(file_path)

    # Step 2: Get chatbot response
    response = rag_chain.invoke({"input": user_text})
    answer = response["answer"]

    # Step 3: Crisis detection
    suicidal_keywords = ["suicide", "kill myself", "end my life", "not worth living", "die"]
    if any(word in user_text.lower() for word in suicidal_keywords):
        trigger_emergency_call()
        answer += (
            "\n\n⚠ This may be a crisis. I've triggered an emergency alert. "
            "Please also reach out to a helpline: India: 9152987821 | US: 988."
        )

    # Step 4: Convert answer to speech
    audio_path = text_to_speech(answer)

    return jsonify({"text": answer, "audio_url": f"/play_audio?path={audio_path}"})

@app.route("/play_audio")
def play_audio():
    path = request.args.get("path")
    return send_file(path, mimetype="audio/mpeg")


if __name__ == '__main__':
   app.run(host="0.0.0.0", port= 8080, debug= True)