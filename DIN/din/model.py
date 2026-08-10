import torch
import torch.nn as nn

class MyDIN(nn.Module):
    def __init__(self, user_count, item_count, cate_count, cate_list, hidden_dim=128):
        """
        Args:
            user_count: user count
            item_count: item count
            cate_count: category count
            cate_list : category list, cate_list[i] is the category count of item i
        """
        super().__init__()
        self.u_embedding = nn.Embedding(user_count, hidden_dim)
        self.i_embedding = nn.Embedding(item_count, hidden_dim)
        self.c_embedding = nn.Embedding(cate_count, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 80),
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1)
        )
        
    def forward(self, x):
        u, i, c, hist_i = x
        u_emb = self.u_embedding(u)
        i_emb = self.i_embedding(i)
        c_emb = self.c_embedding(c)
        x = torch.cat([u_emb, i_emb, c_emb], dim=1)
        return x
        