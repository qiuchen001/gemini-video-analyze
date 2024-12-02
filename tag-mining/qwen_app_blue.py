from flask import Flask
from qwen.upload import upload_bp
from qwen.mining import mining_bp

app = Flask(__name__)

app.register_blueprint(upload_bp)
app.register_blueprint(mining_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30500, debug=True)