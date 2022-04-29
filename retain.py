import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlphaAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.a_att = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        return torch.softmax(self.a_att(g), 1)


class BetaAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.b_att = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        return torch.tanh(self.b_att(h))


def attention_sum(alpha, beta, rev_v, rev_masks):
    s_masks = torch.sum(rev_masks, 2)
    s_masks[s_masks > 0] = 1
    s_masks = torch.unsqueeze(s_masks, 2)
    s_masks = s_masks.float()
    v = torch.mul(s_masks, rev_v)
    comb = torch.mul(torch.mul(alpha, beta), v)
    c = torch.sum(comb, 1)
    return c


def sum_embeddings_with_mask(x, masks):
    x = x * masks.unsqueeze(-1)
    x = torch.sum(x, dim=-2)
    return x


class RETAIN(nn.Module):
    def __init__(self, num_codes, num_categories, embedding_dim=128):
        super().__init__()
        self.num_codes = num_codes
        self.num_categories = num_categories
        self.emb_dim = embedding_dim

        # Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
        self.embedding = nn.Embedding(self.num_codes, self.emb_dim)
        # Define the RNN-alpha using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_a = nn.GRU(self.emb_dim, self.emb_dim, batch_first=True)
        # Define the RNN-beta using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_b = nn.GRU(self.emb_dim, self.emb_dim, batch_first=True)
        # Define the alpha-attention using `AlphaAttention()`;
        self.att_a = AlphaAttention(self.emb_dim)
        # Define the beta-attention using `BetaAttention()`;
        self.att_b = BetaAttention(self.emb_dim)
        # Define the linear layers using `nn.Linear()`;
        self.fc = nn.Linear(self.emb_dim, self.num_categories)
        # Define the final activation layer using `nn.Sigmoid().
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks, rev_x, rev_masks):
        """
        Arguments:
            rev_x: the diagnosis sequence in reversed time of shape (# visits, batch_size, # diagnosis codes)
            rev_masks: the padding masks in reversed time of shape (# visits, batch_size, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        # 1. Pass the reversed sequence through the embedding layer;
        rev_x = self.embedding(rev_x)
        # 2. Sum the reversed embeddings for each diagnosis code up for a visit of a patient.
        rev_x = sum_embeddings_with_mask(rev_x, rev_masks)
        # 3. Pass the reversed embeddings through the RNN-alpha and RNN-beta layer separately;
        g, _ = self.rnn_a(rev_x)
        h, _ = self.rnn_b(rev_x)
        # 4. Obtain the alpha and beta attentions using `AlphaAttention()` and `BetaAttention()`;
        alpha = self.att_a(g)
        beta = self.att_b(h)
        # 5. Sum the attention up using `attention_sum()`;
        c = attention_sum(alpha, beta, rev_x, rev_masks)
        # 6. Pass the context vector through the linear and activation layers.
        logits = self.fc(c)
        probs = self.sigmoid(logits)
        return probs.squeeze()



