import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import numpy as np

# Utilisez le bon identifiant de modèle ici
MODEL_NAME = 'MokaExpress/flaubert-french-difficulty'
CLASSIFIER_PATH = 'svm_clf.pkl'

@joblib.memory.cache
def load_model_and_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()  # Important pour mettre le modèle en mode d'évaluation
    classifier = joblib.load(CLASSIFIER_PATH)  # Assurez-vous que le chemin est correct
    return model, tokenizer, classifier

@joblib.memory.cache
def embed_flaubert(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy()  # Convertir en NumPy et déplacer de GPU à CPU si nécessaire

@joblib.memory.cache
def predict_french_difficulty(sentence, model, tokenizer, classifier):
    sentence_embedding = embed_flaubert(sentence, model, tokenizer)
    # Assurez-vous que c'est un tableau 2D
    sentence_embedding_np = sentence_embedding.reshape(1, -1)
    difficulty_prediction = classifier.predict(sentence_embedding_np)[0]
    return difficulty_prediction

# Exemple d'utilisation
model, tokenizer, classifier = load_model_and_classifier()
example_sentence = "Ceci est une phrase simple"
predicted_difficulty = predict_french_difficulty(example_sentence, model, tokenizer, classifier)
print(f"Predicted difficulty level for the sentence: {predicted_difficulty}")