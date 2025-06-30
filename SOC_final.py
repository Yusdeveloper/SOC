import nltk
nltk.download('treebank')
nltk.download('brown')
nltk.download('conll2000')

from nltk.corpus import treebank, brown, conll2000

# Combine tagged sentences from multiple corpora using the universal tagset
combined_tagged_sentences = (
    treebank.tagged_sents(tagset='universal') +
    brown.tagged_sents(tagset='universal') +
    conll2000.tagged_sents(tagset='universal')
)

# Separate words and tags into respective lists
X_data, y_data = [], []
for sentence in combined_tagged_sentences:
    tokens, tags = zip(*sentence)
    X_data.append(list(tokens))
    y_data.append(list(tags))

# Split data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

# Tokenization and sequence padding
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

word_tokenizer = Tokenizer(oov_token='<OOV>')
word_tokenizer.fit_on_texts(X_train)
X_train_sequences = word_tokenizer.texts_to_sequences(X_train)

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(y_train)
y_train_sequences = tag_tokenizer.texts_to_sequences(y_train)

# Define max length and pad all sequences
MAXLEN = 170
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAXLEN, padding='pre')
y_train_padded = pad_sequences(y_train_sequences, maxlen=MAXLEN, padding='pre')

# PyTorch Dataset and DataLoader setup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class POSTaggingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_data = POSTaggingDataset(X_train_padded, y_train_padded)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Define model parameters
vocab_size = len(word_tokenizer.word_index) + 1
num_tags = len(tag_tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 128

# Define the POS Tagger model
class BiLSTMPOSTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(BiLSTMPOSTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        mask = x != 0
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        tag_scores = self.fc(lstm_out)
        return tag_scores, mask

# Set device and training configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMPOSTagger(vocab_size, embedding_dim, hidden_dim, num_tags).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for ep in range(epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        predictions, _ = model(inputs)
        loss = criterion(predictions.view(-1, num_tags), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {ep + 1}, Loss: {epoch_loss:.4f}")

# Inference function to tag sentences
def tag_input_sentences(sent_list):
    model.eval()
    sequences = word_tokenizer.texts_to_sequences(sent_list)
    padded_inputs = pad_sequences(sequences, maxlen=MAXLEN, padding='pre')
    input_tensor = torch.LongTensor(padded_inputs).to(device)

    with torch.no_grad():
        outputs, _ = model(input_tensor)
        predicted = torch.argmax(outputs, dim=-1).cpu().numpy()

    results = []
    for i, seq in enumerate(sequences):
        tags_seq = predicted[i][-len(seq):]
        words = [word_tokenizer.index_word.get(idx, '<UNK>') for idx in seq]
        tags = [tag_tokenizer.index_word.get(tag, 'X') for tag in tags_seq]
        results.append(list(zip(words, tags)))

    return results

# Sample sentences for testing
examples = [
    "Brown refused to testify.",
    "Come as you are"
]
tagged_output = tag_input_sentences(examples)
for tagged_sentence in tagged_output:
    print(tagged_sentence)
