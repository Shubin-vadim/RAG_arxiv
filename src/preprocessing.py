import re

def preprocessing(df):
    pass

def cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]','', text)
    text = re.sub(r'\s+',' ', text)
    return text
