from flask import Blueprint, render_template, request, jsonify
from .sentiment_analysis import get_sentiment

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    prediction = get_sentiment(text)
    return jsonify({'result': prediction})
