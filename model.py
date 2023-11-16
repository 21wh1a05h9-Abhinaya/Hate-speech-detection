import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import string
import nltk
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r"\@w+|\#", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    tweet_tokens = word_tokenize(text)
    filtered_tweets = [w for w in tweet_tokens if not w in stopword] 
    return " ".join(filtered_tweets)

data = pd.read_csv("labeled_data.csv")
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]

stopword = stopwords.words('english')

data.tweet = data['tweet'].apply(clean)
tweetData = data.drop_duplicates("tweet")

lemmatizer = WordNetLemmatizer()
tweetData.loc[:, 'tweet'] = tweetData['tweet'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

tweetData['labels'].value_counts()
non_hate_tweets = tweetData[tweetData.labels == 'No Hate and Offensive']

text = ''.join([word for word in non_hate_tweets['tweet']])
vect = TfidfVectorizer(ngram_range=(1, 3)).fit(tweetData['tweet'])

X = vect.transform(tweetData['tweet'])
Y = tweetData['labels']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, Y_train)

with open("model.pkl", "wb") as model_file:
    pickle.dump(svm_model, model_file)
with open("vect.pkl", "wb") as vect_file:
    pickle.dump(vect, vect_file)