from flask import Flask
from qwen.upload import upload_bp
from qwen.mining import mining_bp
from qwen.summary import summary_bp
from qwen.embedding_summary import embedding_summary_bp

app = Flask(__name__)

app.register_blueprint(upload_bp)
app.register_blueprint(mining_bp)
app.register_blueprint(summary_bp)
app.register_blueprint(embedding_summary_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30500, debug=True)