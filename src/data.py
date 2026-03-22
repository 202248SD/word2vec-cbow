import nltk

def clean_words(words : list[str]) -> list[str]:
    cleaned_words = [word.lower() for word in words]
    return sorted(list(set(cleaned_words)))

def clean_sents(sents : list[list[str]]) -> list[list[str]]:
    return [[w.lower() for w in sent] for sent in sents]

def one_hot_convert(words : list[str]) -> dict[str, int]:
    one_hot_encoding = dict()
    for i, word in enumerate(words):
        one_hot_encoding[word] = i
    return one_hot_encoding

def preprocess_data(corpus : nltk.corpus, window : int) -> tuple[list[str], list[tuple[list[str], str]]]:
    vocab = clean_words(corpus.words())
    sents = clean_sents(corpus.sents())

    min_length = 2*window+1
    data = []

    for sent in sents:
        L = len(sent)

        if L < min_length:
            continue

        for i in range(L-min_length+1):
            window_words = sent[i:i+min_length]
            target = window_words[window]
            context = window_words[:window]+window_words[window+1:]

            data.append((context, target))
    return vocab, data