import numpy as np
from collections import Counter
from codecs import open

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

def train_nb(documents, labels, alpha=1.0):
    class_counts = Counter(labels)
    total_docs = len(labels)
    
    log_priors = {c: np.log(count / total_docs) for c, count in class_counts.items()}
    
    word_counts = {c: Counter() for c in class_counts}
    total_words = {c: 0 for c in class_counts}
    vocab = set()
    
    for doc, label in zip(documents, labels):
        word_counts[label].update(doc)
        total_words[label] += len(doc)
        vocab.update(doc)
        
    vocab_size = len(vocab)
    
    log_likelihoods = {c: {} for c in class_counts}
    default_log_likelihoods = {} 
    
    for c in class_counts:
        denominator = total_words[c] + alpha * vocab_size
        default_log_likelihoods[c] = np.log(alpha / denominator)
        for word in vocab:
            count = word_counts[c].get(word, 0)
            log_likelihoods[c][word] = np.log((count + alpha) / denominator)
            
    return {
        'log_priors': log_priors,
        'log_likelihoods': log_likelihoods,
        'default_log_likelihoods': default_log_likelihoods,
        'classes': list(class_counts.keys())
    }

def score_doc_label(document, label, model):
    score = model['log_priors'][label]
    for word in document:
        score += model['log_likelihoods'][label].get(word, model['default_log_likelihoods'][label])
    return score

def classify_nb(document, model):
    scores = {c: score_doc_label(document, c, model) for c in model['classes']}
    return max(scores, key=scores.get)

def classify_documents(docs, model):
    return [classify_nb(doc, model) for doc in docs]

def accuracy(true_labels, guessed_labels):
    correct = sum(1 for true, guess in zip(true_labels, guessed_labels) if true == guess)
    return correct / len(true_labels)

def get_short_misclassified(docs, true_labels, guessed_labels, model, max_len=25, limit=3):
    errors = []
    for doc, true_l, guess_l in zip(docs, true_labels, guessed_labels):
        if true_l != guess_l and len(doc) <= max_len:
            pos_score = score_doc_label(doc, 'pos', model)
            neg_score = score_doc_label(doc, 'neg', model)
            margin = abs(pos_score - neg_score)
            errors.append((margin, true_l, guess_l, pos_score, neg_score, " ".join(doc)))
    errors.sort(reverse=True, key=lambda x: x[0])
    return errors[:limit]

if __name__ == "__main__":
    file_path = 'all_sentiment_shuffled.txt'
    
    all_docs, all_labels = read_documents(file_path)
    
    split_point = int(0.80 * len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]
    
    trained_model = train_nb(train_docs, train_labels, alpha=1.0)
    predictions = classify_documents(eval_docs, trained_model)
    
    acc = accuracy(eval_labels, predictions)
    print(f"Classifier Accuracy: {acc * 100:.2f}%\n")

    for sample in [['great'], ['bad'], ['a', 'top-quality', 'performance']]:
        p_s = score_doc_label(sample, 'pos', trained_model)
        n_s = score_doc_label(sample, 'neg', trained_model)
        print(f"Sample {sample}: pos={p_s:.2f}, neg={n_s:.2f}, pred={classify_nb(sample, trained_model)}")

    print("\nTop Short Misclassifications:")
    for margin, t_l, g_l, p_s, n_s, text in get_short_misclassified(eval_docs, eval_labels, predictions, trained_model):
        print(f"[True: {t_l} | Pred: {g_l} | Pos: {p_s:.2f}, Neg: {n_s:.2f}] -> \"{text}\"")