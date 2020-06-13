import os


def read(text):
    if os.path.isfile(text):
        return open(text, 'r').readline()
    else:
        return text
