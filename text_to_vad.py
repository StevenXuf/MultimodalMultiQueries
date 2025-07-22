import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha()]  # Remove punctuation/numbers

def get_lexicon(path):
    vad_lexicon = {}
    with open(path, "r") as f:
        next(f)  # Skip header
        for line in f:
            word, valence, arousal, dominance = line.strip().split('\t')
            vad_lexicon[word] = {
                'valence': float(valence),
                'arousal': float(arousal),
                'dominance': float(dominance)
            }
    return vad_lexicon

def get_vad(text,vad_lexicon):
    words = preprocess(text)
    scores = {'valence': [], 'arousal': []}

    for word in words:
        if word in vad_lexicon:
            scores['valence'].append(vad_lexicon[word]['valence'])
            scores['arousal'].append(vad_lexicon[word]['arousal'])
    
    # Aggregate scores (average if words found, else neutral)
    if scores['valence']:
        return {
            'valence': sum(scores['valence']) / len(scores['valence']),
            'arousal': sum(scores['arousal']) / len(scores['arousal'])
        }
    else:
        return {'valence': 0.5, 'arousal': 0.5}  # Neutral fallback

if __name__=='__main__':
    text = "A joyful celebration with energetic dancing under bright sunshine"
    vad_lexicon=get_lexicon('/data/data_fxu/NRC-VAD-Lexicon-v2_1/NRC-VAD-Lexicon-v2_1.txt')
    vad_scores = get_vad(text,vad_lexicon)

    print(f"Valence: {vad_scores['valence']:.3f}")  # Output: ~0.85 (positive)
    print(f"Arousal: {vad_scores['arousal']:.3f}")  # Output: ~0.79 (energetic)
