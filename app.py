import torch
import torch.nn as nn
import pickle
from flask import Flask, render_template, request
import re

# --- 1. MODEL DEFINITIONS (Keep these exactly as they are) ---
class Skipgram(nn.Module):
    def __init__(self, voc_size, emb_size):
        super().__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)

class SkipgramNeg(nn.Module):
    def __init__(self, voc_size, emb_size):
        super().__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid = nn.LogSigmoid()

class Glove(nn.Module):
    def __init__(self, voc_size, emb_size):
        super().__init__()
        self.wi, self.wj = nn.Embedding(voc_size, emb_size), nn.Embedding(voc_size, emb_size)
        self.bi, self.bj = nn.Embedding(voc_size, 1), nn.Embedding(voc_size, 1)

# --- 2. LOAD DATA & MODELS ---
with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)

word2index, corpus_raw, corpus_tokens = data['word2index'], data['corpus_raw'], data['corpus_tokens']
voc_size, emb_size = data['voc_size'], data['emb_size']
index2word = {idx: v for v, idx in word2index.items()}


models = {
    'sg': Skipgram(voc_size, emb_size),
    'neg': SkipgramNeg(voc_size, emb_size),
    'glove': Glove(voc_size, emb_size)
}
models['sg'].load_state_dict(torch.load('model_sg.pth', map_location=torch.device('cpu'))) #important
models['neg'].load_state_dict(torch.load('model_neg.pth', map_location=torch.device('cpu'))) #important
models['glove'].load_state_dict(torch.load('model_glove.pth', map_location=torch.device('cpu')))#important

for m in models.values():
    m.eval()
    for param in m.parameters(): # Add this to freeze the parameters
        param.requires_grad = False

# --- 3. UPDATED VECTOR UTILITIES (With Normalization) ---
def get_word_vec(word, model_type):
    m = models[model_type]
    # Check if word exists, otherwise use <UNK>
    idx_val = word2index.get(word.lower(), word2index['<UNK>'])
    idx = torch.LongTensor([idx_val])

    with torch.no_grad():
        if model_type == 'glove':
            v = (m.wi(idx) + m.wj(idx)) / 2
        else:
            v = (m.embedding_center(idx) + m.embedding_outside(idx)) / 2

    # NORMALIZATION: Scale vector to length 1
    v = v.squeeze()
    norm = v.norm(p=2)
    if norm == 0:  # Handle zero vectors
        return torch.zeros(emb_size)  # or some other default vector
    return v / (norm + 1e-9) # ADDED small number
def detokenize(tokens):
    return ' '.join(tokens)

def get_doc_vectors(model_type):
    doc_vecs = []
    for tokens in corpus_tokens:
        vecs = [get_word_vec(w, model_type) for w in tokens if w in word2index]
        if vecs:
            # Average the word vectors
            doc_v = torch.mean(torch.stack(vecs), dim=0)
            # NORMALIZATION: Scale doc vector to length 1
            norm = doc_v.norm(p=2)
            if norm == 0:
                doc_vec = torch.zeros(emb_size)
            else:
                doc_vec = doc_v / (norm + 1e-9)
            doc_vecs.append(doc_vec)
        else:
            doc_vecs.append(torch.zeros(emb_size))  # zero vector for empty documents
    return torch.stack(doc_vecs)

# Pre-calculate normalized sets
all_doc_vectors = {
    'sg': get_doc_vectors('sg'),
    'neg': get_doc_vectors('neg'),
    'glove': get_doc_vectors('glove')
}

# --- 4. FLASK APP ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', selected_model='sg')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query').strip()
    model_type = request.form.get('model_type', 'sg')

    # 1. Handle case where word isn't in vocabulary
    is_unknown = query.lower() not in word2index

    # 2. Get Query Vector (Already normalized in utility)
    query_vec = get_word_vec(query, model_type)

    # 3. Get Doc Vectors (Already normalized)
    doc_vectors = all_doc_vectors[model_type]

    # 4. TASK REQUIREMENT: Compute Dot Product
    # Because vectors are normalized, dot product = cosine similarity
    scores = torch.matmul(doc_vectors, query_vec)

    # 5. Get top 10
    top_scores, top_indices = torch.topk(scores, k=min(10, len(scores)))

    results = []
    for j, i in enumerate(top_indices):
        original_tokens = corpus_tokens[i.item()]
        # Corrected Part: Display original text
        results.append({
            "text": detokenize(original_tokens),  #corrected Part
            "score": f"{top_scores[j].item():.4f}"
        })

    return render_template('index.html',
                           query=query,
                           results=results,
                           selected_model=model_type,
                           is_unknown=is_unknown)

if __name__ == '__main__':
    app.run(debug=True)