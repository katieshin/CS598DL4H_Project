import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, num_codes, num_categories, emb_dim):
        super().__init__()
        """
        TODO: 
            1. Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
            2. Define the RNN using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
            2. Define the RNN for the reverse direction using `nn.GRU()`;
               Set `hidden_size` to 128. Set `batch_first` to True.
            3. Define the linear layers using `nn.Linear()`; Set `in_features` to 256, and `out_features` to 1.
            4. Define the final activation layer using `nn.Sigmoid().

        Arguments:
            num_codes: total number of diagnosis codes
        """
        self.num_codes = num_codes
        self.num_categories = num_categories
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(self.num_codes, self.emb_dim)
        # self.rnn = nn.GRU(emb_dim, hidden_size=emb_dim, batch_first=True)
        # self.rev_rnn = nn.GRU(emb_dim, hidden_size=emb_dim, batch_first=True)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size=self.emb_dim, batch_first=True)
        self.rev_lstm = nn.LSTM(self.emb_dim, hidden_size=self.emb_dim, batch_first=True)
        self.fc = nn.Linear(self.emb_dim, self.num_categories)
        self.sigmoid = nn.Sigmoid()

    def sum_embeddings_with_mask(self, x, masks):
        x[masks == 0] = 0
        out = torch.sum(x, 2)
        return out

    def get_last_visit(self, hidden_states, masks):
        s_masks = torch.sum(masks, 2)
        s_masks[s_masks > 0] = 1
        z_masks = torch.sum(s_masks, 1)-1
        # REPLACE if not using mac: mask = torch.zeros(*hidden_states.shape[:2]).cuda()
        mask = torch.zeros(*hidden_states.shape[:2]).to(device)
        # mask.zero_()
        mask.scatter_(1, z_masks.view(-1, 1), 1)
        states = hidden_states.clone()
        states[mask == 0] = 0
        out = torch.sum(states, 1)
        return out

    def forward(self, x, masks):
        """
        Arguments:
            x: the diagnosis sequence of shape (batch_size, # visits, # diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        # 1. Pass the sequence through the embedding layer;
        x = self.embedding(x)
        # 2. Sum the embeddings for each diagnosis code up for a visit of a patient.
        x = self.sum_embeddings_with_mask(x, masks)

        # 3. Pass the embeddings through the RNN layer;
        output, _ = self.lstm(x)

        # REPLACE if not using mac: logits = self.fc(self.get_last_visit(output.cuda(), masks.cuda()))
        logits = self.fc(self.get_last_visit(output.to(device), masks.to(device)))
        probs = self.sigmoid(logits)
        # print(probs.shape)
        return probs