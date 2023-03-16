import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import classification_report

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.transitions = nn.Parameter(torch.randn(len(tag_to_ix), len(tag_to_ix)))
        self.transitions.data[tag_to_ix['<START>'], :] = -10000
        self.transitions.data[:, tag_to_ix['<STOP>']] = -10000
        
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix['<START>']] = 0.
        forward_var = init_alphas
        
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']]
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha
    
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix['<START>']], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix['<STOP>'], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix['<START>']] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']]
            best_tag_id = torch.argmax(terminal_var).tolist()
            path_score = terminal_var[0][best_tag_id]
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            assert start == self.tag_to_ix['<START>']
            best_path.reverse()
            return path_score, best_path

        def _get_lstm_features(self, sentence):
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
            lstm_feats = self.hidden2tag(lstm_out)
            return lstm_feats

        def neg_log_likelihood(self, sentence, tags):
            self.tagset_size = len(self.tag_to_ix)
            lstm_feats = self._get_lstm_features(sentence)
            forward_score = self._forward_alg(lstm_feats)
            gold_score = self._score_sentence(lstm_feats, tags)
            return forward_score - gold_score

        def forward(self, sentence):
            self.tagset_size = len(self.tag_to_ix)
            lstm_feats = self._get_lstm_features(sentence)
            score, tag_seq = self._viterbi_decode(lstm_feats)
            return score, tag_seq

class TextDataset(Dataset):
    def init(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def pad_collate(batch):
    X_batch, y_batch = zip(*batch)
    X_batch = pad_sequence(X_batch, batch_first=True)
    y_batch = pad_sequence(y_batch, batch_first=True)
    return X_batch, y_batch

def train(model, optimizer, train_loader, device):
    model.train()
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        loss = model.neg_log_likelihood(X_batch, y_batch)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"Step {i} Loss: {loss.item()}")

def evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            _, y_pred_batch = model(X_batch)
            y_true.extend(y_batch.view(-1).tolist())
            y_pred.extend(y_pred_batch)
            print(classification_report(y_true, y_pred))

def main():
    tag_to_ix = {'O': 0, 'B': 1, 'I': 2, '<START>': 3, '<STOP>': 4}
    vocab_size = 10000
    embedding_dim = 100
    hidden_dim = 128
    model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim, hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

    test_dataset = TextDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        train(model, optimizer, train_loader, device)
        evaluate(model, test_loader, device)