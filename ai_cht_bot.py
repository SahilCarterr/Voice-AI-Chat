import gradio as gr
import openai
import os
import tempfile

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=f
        )
    return transcript.text

def generate_response(conversation_history):
    response = openai.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::BA52Cq4i",
        messages=conversation_history
    )
    reply = response.choices[0].message.content
    return reply

def text_to_speech(text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    temp_audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    response.stream_to_file(temp_audio_path)
    return temp_audio_path

def process_input(audio, text, conversation_history):
    if not conversation_history:
        conversation_history = [{"role": "system", "content": "You are a pilot assistant."}]
    transcribed_text = ""
    if audio:
        transcribed_text = transcribe_audio(audio)
    else:
        transcribed_text = text or ""
    if not transcribed_text.strip():
        return (
            "No text found. Please provide audio or type text.",
            "Please provide text or record audio.",
            None,
            conversation_history
        )
    conversation_history.append({"role": "user", "content": transcribed_text})
    generated_text = generate_response(conversation_history)
    conversation_history.append({"role": "assistant", "content": generated_text})
    audio_output = None
    try:
        audio_output = text_to_speech(generated_text)
    except Exception:
        pass
    return transcribed_text, generated_text, audio_output, conversation_history

def update_chat_history(conversation_history):
    return "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history
    )

with gr.Blocks() as demo:
    gr.Markdown("Voice AI Chat")
    user_name = gr.Textbox(label="Enter your name")

    with gr.Row():
        audio_input = gr.Audio(
            sources="microphone", 
            type="filepath", 
            label="Record Audio"
        )
        text_input = gr.Textbox(
            label="Or Type Text",
            placeholder="Enter text here..."
        )
    
    generate_btn = gr.Button("Generate Response")
    conversation_history = gr.State([])

    transcribed_textbox = gr.Textbox(
        label="Transcribed Speech-to-Text", 
        interactive=False, 
        lines=1
    )
    output_text = gr.Textbox(
        label="GPT Response", 
        interactive=False, 
        lines=2
    )
    output_audio = gr.Audio(
        label="Generated Audio", 
        autoplay=True, 
        interactive=False
    )
    chat_history_display = gr.Textbox(
        label="Conversation History", 
        interactive=False, 
        lines=3
    )

    generate_btn.click(
        fn=process_input,
        inputs=[audio_input, text_input, conversation_history],
        outputs=[transcribed_textbox, output_text, output_audio, conversation_history]
    ).then(
        fn=update_chat_history,
        inputs=[conversation_history],
        outputs=[chat_history_display]
    )

    end_conversation = gr.Button("End Conversation")
    end_conversation.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="() => { window.location.reload() }"
    )

demo.launch()