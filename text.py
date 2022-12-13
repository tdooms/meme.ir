import re


def clean_text(text):
    new_text = text.replace('\n', ' ')  # remove new lines
    new_text = new_text.encode('ascii', 'ignore').decode()  # remove non-ascii
    new_text = re.sub('\s+', ' ', new_text)  # remove long spaces
    new_text = re.sub('\s$', '', new_text)  # remove trailing space
    splitted_text = new_text.split(' ')  # split text into words
    return new_text
