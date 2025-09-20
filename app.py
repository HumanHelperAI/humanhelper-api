from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status":"ok","message":"Human Helper API is live"})

# example endpoint
@app.route("/echo", methods=["POST"])
def echo():
    data = request.json or {}
    return jsonify({"you_sent": data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
