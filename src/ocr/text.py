import re
import enchant
import distance
import numpy as np


def hamming_dist(word1, word2, penalize_length=True):
    word1 = word1.lower()
    word2 = word2.lower()
    min_len = min(len(word1), len(word2))
    # calculate hamming distance
    dist = distance.hamming(word1[:min_len], word2[:min_len])

    # also penalize on length difference
    if penalize_length:
        dist += abs(len(word1) - len(word2))

    return dist


def get_shortest_hamming_dist(words, other_word):
    if len(words) == 0:
        return np.inf
    words_dist = map(lambda word: hamming_dist(word, other_word), words)
    return min(words_dist)


def get_shortest_hamming_dist_word(words, other_word):
    words_dist = map(lambda word: hamming_dist(word, other_word), words)
    return words[np.argmin(list(words_dist))]


def get_dictionary():
    eng_dict = enchant.Dict("en_US")
    eng_dict.add('OpenAI')
    eng_dict.add('ChatGPT')
    eng_dict.add('fruitloops')
    eng_dict.add('GPT')
    eng_dict.add('GPT-2')
    eng_dict.add('GPT-3')
    eng_dict.add('imgflip')
    return eng_dict


def clean_text(text):
    new_text = text.replace('\n', ' ')  # remove new lines
    new_text = re.sub('\s+', ' ', new_text)  # remove long spaces
    new_text = re.sub('\s$', '', new_text)  # remove trailing space
    new_text = re.sub('^\s', '', new_text)  # remove leading space
    # add space between number and word
    new_text = re.sub(r'([0-9]+)([a-zA-Z]+)', r'\1 \2', new_text)
    words = new_text.split(' ')  # split text into words
    valid_words = list()
    skip_word = 0
    for word_idx, word in enumerate(words):
        if skip_word:
            skip_word -= 1
        elif not len(word):
            continue
        elif len(word) == 1 and word.lower() not in ['a', 'i', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            continue
        elif eng_dict.check(word):
            valid_words.append(word)
        # if word is at end of sentence or has a period, check if it's valid
        elif eng_dict.check(word[:-1]) and word[-1] in ['.', '!', '?', ':', ';', ',', ')', ']', '}']:
            valid_words.append(word)
        elif eng_dict.check(word[1:]) and word[0] in ['"', "'", '(', '[', '{']:
            valid_words.append(word)
        # check if word is a number
        elif re.match('^\d+$', word):
            valid_words.append(word)
        # check if word is a number with a comma
        elif re.match('^\d+,\d+$', word):
            valid_words.append(word)
        # check if word is a number with a comma and a dot
        elif re.match('^\d+,\d+\.\d+$', word):
            valid_words.append(word)
        # check if word is a number with a dot
        elif re.match('^\d+\.\d+$', word):
            valid_words.append(word)
        # check if word is a number with a dot and a comma
        elif re.match('^\d+\.\d+,\d+$', word):
            valid_words.append(word)
        # check if there is a word close to the suggested if so then use word
        elif get_shortest_hamming_dist(suggestions := eng_dict.suggest(word),
                                       word) < 3:
            valid_words.append(word)
        # check if subsequent word completes the word
        elif word_idx + 1 < len(words):
            new_word = word
            broken = False
            for i in range(word_idx + 1, len(words)):
                new_word += words[i]
                skip_word += 1
                if eng_dict.check(new_word):
                    valid_words.append(new_word)
                    broken = True
                    break
            if not broken:
                skip_word = 0
    new_text = ' '.join(valid_words)

    return new_text


eng_dict = get_dictionary()

if __name__ == '__main__':
    text = 'Days before OpenAL'
    print(clean_text(text))
