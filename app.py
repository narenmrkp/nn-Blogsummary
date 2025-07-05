import os, requests
import gradio as gr
from groq import Groq
from elevenlabs import ElevenLabs, save
from datetime import datetime

# 🔐 Read API keys from environment (Render uses this)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def get_web_research(topic):
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": topic, "search_depth": "advanced", "include_answer": True}
    res = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
    results = res.json().get("results", [])
    return "\n\n".join([r.get("content", "")[:3000] for r in results])

def generate_script(topic, research):
    prompt = f"""
You are a podcast scriptwriter. Based on the topic and research below, generate:
1. Title
2. 4-point outline
3. A 2-minute engaging script

== Topic ==
{topic}

== Research ==
{research}
"""
    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()

def generate_audio(script_text):
    audio = eleven_client.text_to_speech.convert(
        text=script_text,
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_monolingual_v1",
        output_format="mp3_44100_128"
    )
    fname = f"podcast_{datetime.now().strftime('%H%M%S')}.mp3"
    with open(fname, "wb") as f: f.write(audio)
    return fname

def pipeline(topic):
    try:
        research = get_web_research(topic)
        script = generate_script(topic, research)
        audio_file = generate_audio(script)
        return script, audio_file
    except Exception as e:
        return f"❌ Error: {e}", None

# 🎛 Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 🎙️ AI Podcast Generator (Groq + Tavily + ElevenLabs)")
    topic = gr.Textbox(label="🎯 Topic")
    script = gr.Textbox(label="📝 Script", lines=15)
    audio = gr.Audio(label="🔊 Audio")
    btn = gr.Button("🎧 Generate Podcast")
    btn.click(fn=pipeline, inputs=topic, outputs=[script, audio])
app.launch()
