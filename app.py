from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the saved model
model_classification = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model_classification.load_state_dict(torch.load("model_classification.pth", map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to predict
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model_classification(**inputs)
    predicted_class = torch.argmax(outputs.logits)
    return predicted_class.item()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_review():
    if request.method == 'POST':
        review_text = request.form['review_text']
        predicted_class = predict(review_text)
        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
