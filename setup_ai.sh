#!/data/data/com.termux/files/usr/bin/bash
# One-shot Termux 11X setup for Human Helper.AI & Cherry.AI
# Paste and run in Termux 11X (F-Droid / GitHub release)
set -e
echo "=== START: Human Helper.AI & Cherry.AI Termux Setup ==="

# 1) Basic OS update
echo "[1/14] Updating packages..."
pkg update -y && pkg upgrade -y

# 2) Install core packages
echo "[2/14] Installing core packages..."
pkg install -y git python nodejs clang openjdk-17 wget curl unzip nano vim tmux

# 3) Storage permission (interactive)
echo "[3/14] Granting storage access (termux-setup-storage). Please confirm permission popup on Android."
termux-setup-storage || true
# (termux-setup-storage may require user to accept permission in UI)

# 4) Python packages (pip)
echo "[4/14] Upgrading pip and installing Python libs..."
pip install --upgrade pip setuptools wheel
pip install flask flask-cors requests python-dotenv openai sounddevice websocket-client

# 5) Voice / STT / TTS libs
echo "[5/14] Installing speech libraries (vosk, coqui-tts will be attempted)..."
pip install vosk
# Coqui TTS can be heavy — try install, fallback if fails
pip install --no-cache-dir coqui-tts || echo "coqui-tts install failed or is heavy — continue and install later if needed"

# 6) Termux API and pulseaudio for sound
echo "[6/14] Installing termux-api and pulseaudio..."
pkg install -y termux-api pulseaudio

# 7) Setup pulseaudio configuration for Termux
echo "[7/14] Setting up PulseAudio service (user-level)..."
# create simple start script
cat > ~/start-pulse.sh <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
pulseaudio --start --exit-idle-time=-1
echo "PulseAudio started"
EOF
chmod +x ~/start-pulse.sh

# 8) Install Vosk model helper (not downloading models due to size)
echo "[8/14] Vosk: helper script created. You'll need to download Vosk models separately."
cat > ~/vosk_transcribe.py <<'PY'
import sys, wave, json
from vosk import Model, KaldiRecognizer

if len(sys.argv) < 3:
    print("Usage: python vosk_transcribe.py /path/to/model /path/to/file.wav")
    sys.exit(1)

model_path = sys.argv[1]
wav_path = sys.argv[2]
wf = wave.open(wav_path, "rb")
model = Model(model_path)
rec = KaldiRecognizer(model, wf.getframerate())
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())
print(rec.FinalResult())
PY

# 9) Install llama.cpp (local LLM runtime)
echo "[9/14] Cloning and building llama.cpp (local LLM runner)..."
cd $HOME
if [ -d "$HOME/llama.cpp" ]; then
  echo "llama.cpp already exists — skipping clone (will try to make)."
else
  git clone https://github.com/ggerganov/llama.cpp || { echo "git clone failed"; }
fi
cd llama.cpp || { echo "llama.cpp directory missing"; }
# Build with make; ignore errors but try best-effort
make clean || true
make -j2 || make

# 10) Create a simple wrapper script to run llama.cpp models
cat > ~/run_local_llm.sh <<'LLM'
#!/data/data/com.termux/files/usr/bin/bash
# Usage: ./run_local_llm.sh /sdcard/Download/model.gguf "Hello"
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/model.gguf \"prompt text\""
  exit 1
fi
MODEL="$1"
PROMPT="$2"
cd ~/llama.cpp || exit 1
./main -m "$MODEL" -p "$PROMPT"
LLM
chmod +x ~/run_local_llm.sh

# 11) Basic Flask backend template
echo "[11/14] Creating a simple Flask backend template ~/humanhelper_backend/app.py ..."
mkdir -p ~/humanhelper_backend
cat > ~/humanhelper_backend/app.py <<'PY'
from flask import Flask, request, jsonify
import os
app = Flask(__name__)

@app.route("/")
def index():
    return "Human Helper.AI backend running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    prompt = data.get("prompt","")
    # Basic echo behavior. Replace with OpenAI or local LLM calls.
    return jsonify({"reply": f"Echo: {prompt}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
PY

# 12) Helpful aliases in ~/.profile
echo "[12/14] Adding helpful aliases to ~/.profile ..."
cat >> ~/.profile <<'BASH'

# Human Helper.AI shortcuts
alias startpulse="$HOME/start-pulse.sh"
alias runbackend="python3 ~/humanhelper_backend/app.py"
alias runllm="$HOME/run_local_llm.sh"
alias vosktrans="python3 ~/vosk_transcribe.py"
BASH

# 13) Final notes written to ~/FIRST_RUN_NOTES.txt
echo "[13/14] Writing FIRST_RUN_NOTES.txt with next steps..."
cat > ~/FIRST_RUN_NOTES.txt <<'TXT'
Human Helper.AI & Cherry.AI Termux Setup - NEXT STEPS:

1) Add OpenAI API key (if you want cloud GPT):
   export OPENAI_API_KEY="sk-..."
   Put this in ~/.bashrc or use python-dotenv in your project.

2) Vosk STT:
   - Download a Vosk model (small) and put it under /sdcard/models/vosk-model
   - Example small model (English): https://alphacephei.com/vosk/models
   - To transcribe:
     python3 ~/vosk_transcribe.py /sdcard/models/vosk-model-small-en-us-0.15 /sdcard/Download/sample.wav

3) Coqui TTS:
   - Coqui TTS is heavy. If pip install failed, try in a proot distro or use RHVoice binary.
   - RHVoice can be installed via apt in a proot distro or using packages if available.

4) llama.cpp:
   - Download a quantized ggml/gguf model (large!) and put it in /sdcard/Download/
   - Run: ~/run_local_llm.sh /sdcard/Download/your-model.gguf "Hello Cherry"

5) PulseAudio:
   - Start with: startpulse
   - Then run audio apps that use pulseaudio (TTS playback etc.)

6) Run backend:
   - runbackend
   - Visit http://127.0.0.1:5000 in a browser on phone or use Termux webview.

7) Security:
   - Do NOT hardcode API keys. Use environment variables or .env.
   - Monitor storage & RAM usage. Local LLMs need lots of RAM/disk.

TXT

# 14) Final summary
echo "=== SETUP COMPLETE ==="
echo "Open ~/FIRST_RUN_NOTES.txt for next steps. Reload shell to enable aliases:"
echo "Run: source ~/.profile"
echo "Start PulseAudio: startpulse"
echo "Run backend: runbackend"
echo "Run local LLM: runllm /sdcard/Download/model.gguf \"Hello\""
echo "Vosk transcribe example: vosktrans /sdcard/models/vosk-model /sdcard/Download/sample.wav"
echo "See ~/FIRST_RUN_NOTES.txt for full instructions."
echo "=== END ==="
