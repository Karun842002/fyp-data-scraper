import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tensorflow import keras
from keras.preprocessing import text, sequence
from torch_random_fields.models import LinearChainCRF
from torch_random_fields.models.constants import Inference, Learning

df = pd.read_csv('./datasets/cleaned1.csv')
df.loc[df['truth_value'] == 'tom_ruling_pof', 'truth_value'] = 'meter-false'
df.loc[df['truth_value'] == 'meter-half-true', 'truth_value'] = 'meter-true'
df.loc[df['truth_value'] == 'meter-mostly-true', 'truth_value'] = 'meter-true'
df.loc[df['truth_value'] == 'meter-mostly-false', 'truth_value'] = 'meter-false'
df = df.dropna(subset=['claim'])
df.reset_index(drop=True, inplace=True)
print(df['truth_value'].value_counts())

le = LabelEncoder()
cdf = df.copy()
cdf['truth_value'] = le.fit_transform(cdf['truth_value'])

stemmer = SnowballStemmer("english")
def stemm_text(text):
    return ' '.join([stemmer.stem(w) for w in text.split(' ')])

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])

T = cdf['claim'].str.split(' \n\n---\n\n').str[0]
T = T.str.replace('-',' ').str.replace('[^\w\s]','').str.replace('\n',' ').str.lower()
stop = stopwords.words('english')

T = T.apply(lambda x: ' '.join([y for y in x.split() if not y.isdigit()]))
T = T.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
cdf['claim'] = T
print(cdf.head(10))


X = cdf['claim']
y = cdf['truth_value']
X = X.apply(lambda w: lemmatize_text(w))

torch.manual_seed(42)
device = torch.device("mps" if torch.has_mps else "cpu")
print(device)


def tokenizeAndGenerateSequences(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    #Maximum number of characters in a sentence = 44
    tk = text.Tokenizer(num_words=1000)
    tk.fit_on_texts(xtrain)
    tokenized_train = tk.texts_to_sequences(xtrain)
    X_train = torch.tensor(sequence.pad_sequences(tokenized_train, maxlen=60)).to(device)
    tokenized_test = tk.texts_to_sequences(xtest)
    X_test = torch.tensor(sequence.pad_sequences(tokenized_test, maxlen=60)).to(device)

    # Convert labels to tensors
    y_train = torch.tensor(ytrain.values).float().to(device)
    y_test = torch.tensor(ytest.values).float().to(device)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = tokenizeAndGenerateSequences(cdf['claim'], cdf['truth_value'])
X_train, y_train, X_test, y_test = tokenizeAndGenerateSequences(X, y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape



# Define the model architecture
class BiLSTM(nn.Module):
    def __init__(self, num_words, embed_size, hidden_size, output_size, dropout_rate):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(num_words, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm1 = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.crf = LinearChainCRF(
            output_size,
            low_rank=5,
            learning=Learning.PIECEWISE,
            inference=Inference.VITERBI,
            feature_size=10,
        )

    def forward(self, x):
        feats = self.embedding(x)
        x = self.dropout(feats)
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        pred = self.fc(x)
        x = self.crf(unaries = pred, node_features = feats, targets )
        return x
    

# Train the model
te = 50
acc = []
tracc = []
for e in range(1, te+1):
    ctracc = 0
    model = BiLSTM(num_words=1000, embed_size=60, hidden_size=64, output_size=2, dropout_rate=0.2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(e):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = model(X_train.to(device))
            predictions = (predictions > 0.5).to('cpu').int().squeeze().numpy()
        train_accuracy = metrics.accuracy_score(y_train.to('cpu'), predictions)
        ctracc += train_accuracy

    ctracc /= e

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions > 0.5).to('cpu').int().squeeze().numpy()

    print(f"Total Epochs: {e}, Train Accuracy: {ctracc} Test Accuracy: {metrics.accuracy_score(y_test.to('cpu'), predictions)}")
    acc.append(metrics.accuracy_score(y_test.to('cpu'), predictions))
    tracc.append(ctracc)
#     break
print('Max acc -', max(acc), ' with epochs -', acc.index(max(acc)))
# plt.plot([i for i in range(1, 51)], acc)


