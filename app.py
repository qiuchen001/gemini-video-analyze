from flask import Flask
from tag_mining.qwen.upload import upload_bp
from tag_mining.qwen.mining import mining_bp
from tag_mining.qwen.summary import summary_bp
from tag_mining.qwen.embedding_summary import embedding_summary_bp
from tag_mining.qwen.embedding_summary_retrieval import embedding_summary_retrieval_bp


app = Flask(__name__)

app.register_blueprint(upload_bp)
app.register_blueprint(mining_bp)
app.register_blueprint(summary_bp)
app.register_blueprint(embedding_summary_bp)
app.register_blueprint(embedding_summary_retrieval_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30500, debug=True)