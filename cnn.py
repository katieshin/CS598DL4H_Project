import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, num_codes, num_categories, emb_dim, max_num_codes):
        super(CNN, self).__init__()
        self.num_codes = num_codes
        self.num_categories = num_categories
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(self.num_codes, self.emb_dim)

        self.conv1 = nn.Conv1d(41, self.emb_dim, 3)
        self.conv2 = nn.Conv1d(self.emb_dim, self.emb_dim, 4)
        self.conv3 = nn.Conv1d(self.emb_dim, num_categories, 5)

        self.fc1 = nn.Linear(self.emb_dim-3-4-5+3, 1)

        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def sum_embeddings_with_mask(self, x, masks):
        x[masks == 0] = 0
        out = torch.sum(x, 2)
        return out

    def forward(self, x, masks, rev_x, rev_masks):
        out = self.embedding(x)
        out = self.sum_embeddings_with_mask(out, masks)
        out = self.dropout(F.relu(self.conv1(out)))
        out = self.dropout(F.relu(self.conv2(out)))
        out = self.dropout(F.relu(self.conv3(out)))
        out = F.relu(self.fc1(out).squeeze(2))
        return self.sigmoid(out)