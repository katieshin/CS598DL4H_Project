import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sum_embeddings_with_mask(x, masks):
    x[masks == 0] = 0
    out = torch.sum(x, 2)
    return out


def get_last_visit(hidden_states, masks):
    s_masks = torch.sum(masks, 2)
    s_masks[s_masks > 0] = 1
    z_masks = torch.sum(s_masks, 1)-1
    mask = torch.zeros(*hidden_states.shape[:2]).to(device)
    mask.scatter_(1, z_masks.view(-1, 1), 1)
    states = hidden_states.clone()
    states[mask == 0] = 0
    out = torch.sum(states, 1)
    return out


class DIPOLE(nn.Module):
    def __init__(self, num_codes, num_categories, embedding_dim=128, max_num_codes=39):
        super().__init__()
        self.num_codes = num_codes
        self.num_categories = num_categories
        self.emb_dim = embedding_dim

        self.visit_emb = nn.Linear(max_num_codes, self.emb_dim)
        self.local = nn.Linear(self.emb_dim, self.emb_dim)

        self.fc1 = nn.Linear(self.emb_dim*4, self.emb_dim, bias=False)
        self.fc2 = nn.Linear(self.emb_dim, self.num_categories)

        self.rnn = nn.GRU(self.emb_dim, hidden_size=self.emb_dim, batch_first=True)
        self.rev_rnn = nn.GRU(self.emb_dim, hidden_size=self.emb_dim, batch_first=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, masks, rev_x, rev_masks):
        """
        Arguments:
            x: the diagnosis sequence of shape (batch_size, # visits, # diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """

        batch_size = x.shape[0]

        x = self.relu(self.visit_emb(x.float()))
        rev_x = self.relu(self.visit_emb(rev_x.float()))

        output, _ = self.rnn(x)
        rev_output, _ = self.rev_rnn(rev_x)

        true_h_n = get_last_visit(output, masks)
        true_h_n_rev = get_last_visit(rev_output, rev_masks)

        output = self.softmax(self.local(output))
        rev_output = self.softmax(self.local(rev_output))
        output = get_last_visit(output, masks)
        rev_output = get_last_visit(rev_output, rev_masks)

        comb = torch.cat([output, rev_output], 1)
        # comb = torch.sum(comb, dim=1)
        # print(comb.shape, true_h_n.shape, true_h_n_rev.shape)

        _h_t = torch.cat([comb, true_h_n, true_h_n_rev], 1)
        _h_t = self.fc1(_h_t)
        _h_t = self.tanh(_h_t)

        probs = self.sigmoid(self.fc2(_h_t))
        return probs