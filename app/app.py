from flask import Flask, render_template, request, jsonify
import torch
import pickle
from utils import Skipgram, SkipgramNeg, Glove  
import torch.nn.functional as F

# Load the data
Data = pickle.load(open('data/data.pkl', 'rb'))
vocab = Data['vocab']
word2index = Data['word2index']
voc_size = Data['voc_size']
embed_size = Data['emb_size']

# Load the models
skipgram = Skipgram(voc_size, embed_size)
skipgram.load_state_dict(torch.load('models/skipgram.pth', map_location=torch.device('cpu')))
skipgram.eval()

skipgramNeg = SkipgramNeg(voc_size, embed_size)
skipgramNeg.load_state_dict(torch.load('models/skipgramNEG.pth', map_location=torch.device('cpu')))
skipgramNeg.eval()

glove = Glove(voc_size, embed_size)
glove.load_state_dict(torch.load('models/GloVe.pth', map_location=torch.device('cpu')))
glove.eval()

model_path = 'models/gensim.pkl'
gensim = pickle.load(open(model_path, 'rb'))

app = Flask(__name__, static_url_path='/static')

# for skipgram, skipgramNEG, glove
def get_similar_words(model, user_inputs):

    all_word_vectors = torch.stack([model.get_embed(word) for word in vocab]) #vectorized all the vocabs

    user_inputs = user_inputs # getting user input

    input_vectors = [] 

     # Iterating over each word in user inputs
    for word in user_inputs:
        if word.lower() in vocab:
            input_vectors.append(model.get_embed(word.lower()))
        else:
            input_vectors.append(model.get_embed('<UNK>'))

    # Check if input vectors are not empty
    if input_vectors:
        
        # Initialize result_vector with the first vector
        result_vector = input_vectors[0]

        # Add the rest of the vectors
        for vector in input_vectors[1:]:
            result_vector += vector
    else:
        # Handle the case where input_vectors is empty
        result_vector = torch.zeros_like(all_word_vectors[0])  # Assuming all vectors have the same size

    # Calculate cosine similarities
    cos_sim = F.cosine_similarity(result_vector.unsqueeze(0), all_word_vectors)

    # Get top 10 similar words
    top_indices = torch.argsort(cos_sim, descending=True)[0][:10]
    return [vocab[index.item()] for index in top_indices.view(-1)]

# for gensim
def get_similar_words_gensim(model, user_inputs):

    user_inputs = user_inputs

    input_vectors = []

     # Iterating over each word in user inputs
    for word in user_inputs:
        if word.lower() in gensim:
            input_vectors.append(model.get_vector(word.lower()))
        else:
            input_vectors.append(model.get_vector('unknown'))

    # Check if input vectors are not empty
    if input_vectors:
        
        # Initialize result_vector with the first vector
        result_vector = input_vectors[0]

        # Add the rest of the vectors
        for vector in input_vectors[1:]:
            result_vector = result_vector + vector

    # Calculate cosine similarities
    cos_sim = gensim.most_similar([result_vector])

    # Create a list of the similar words, ignoring the similarity scores
    result = [word for word, _ in cos_sim]

    return result


@app.route('/')
def index():
    if request.method == 'GET':
        return render_template('index.html', query='')

@app.route('/similar_words/skipgram', methods=['POST'])
def similar_words_skipgram():
    input_words = request.json.get('words', [])
    similar_words = get_similar_words(skipgram, input_words)
    return jsonify(similar_words)

@app.route('/similar_words/skipgramneg', methods=['POST'])
def similar_words_skipgramneg():
    input_words = request.json.get('words', [])
    similar_words = get_similar_words(skipgramNeg, input_words)
    return jsonify(similar_words)

@app.route('/similar_words/glove', methods=['POST'])
def similar_words_glove():
    input_words = request.json.get('words', [])
    similar_words = get_similar_words(glove, input_words)
    return jsonify(similar_words)

@app.route('/similar_words/gensim', methods=['POST'])
def similar_words_gensim():
    input_words = request.json.get('words', [])
    similar_words = get_similar_words_gensim(gensim, input_words)
    return jsonify(similar_words)

if __name__ == '__main__':
    app.run(debug=True)
