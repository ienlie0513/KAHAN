import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

class EmbedAttention(nn.Module):

    def __init__(self, att_size):
        super(EmbedAttention, self).__init__()
        self.attn = nn.Linear(att_size, 1)

    def forward(self, input, len_s, total_length):
        att = self.attn(input).squeeze(-1)
        out = self._masked_softmax(att, len_s, total_length).unsqueeze(-1)

        return out

    def _masked_softmax(self, mat, len_s, total_length):
        
        idxes = torch.arange(0, total_length, out=mat.data.new(total_length)).unsqueeze(1)
        mask = (idxes<len_s.unsqueeze(0).to(idxes.device)).float().t()
        
        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(1, True)+0.0001
     
        return exp/sum_exp.expand_as(exp)

class AttentionalBiRNN(nn.Module):
    def __init__(self, inp_size, hid_size, dropout=0, RNN_cell=nn.GRU):
        super(AttentionalBiRNN, self).__init__()

        self.rnn = RNN_cell(inp_size, hid_size, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.lin = nn.Linear(hid_size*2, hid_size*2)
        self.emb_att = EmbedAttention(hid_size*2)
    
    def forward(self, packed_batch, total_length):

        rnn_output, _ = self.rnn(packed_batch)
        enc_output, len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True, total_length=total_length)

        enc_output = self.dropout(enc_output)
        emb_h = torch.tanh(self.lin(enc_output))

        attended = self.emb_att(emb_h, len_s, total_length) * enc_output
        
        return attended.sum(1)


# news hierarchical attention network
class NHAN(nn.Module):

    def __init__(self, word2vec, emb_size=100, hid_size=100, max_sent=50, dropout=0.3):
        super(NHAN, self).__init__()

        self.max_sent = max_sent
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec.vectors))
        self.word = AttentionalBiRNN(emb_size, hid_size, dropout=dropout)
        self.sent = AttentionalBiRNN(hid_size*2+hid_size//2, hid_size, dropout=dropout)

        self.NEE_attn = nn.MultiheadAttention(hid_size*2, 4)
        self.ent_lin = nn.Linear(hid_size*2, hid_size//2)
        self.relu = nn.ReLU()

    def _reorder_input(self, input, ln, ls):
        # (batch, max_sentence, max_length) to (# of sentences in the batch, max_length)
        reorder_input = [news[:ln[i]] for i, news in enumerate(input)]
        reorder_input = torch.cat(reorder_input, axis=0)

        reorder_ls = [j for i, l in enumerate(ln) for j in ls[i][:l]]

        return reorder_input, reorder_ls

    def _reorder_word_output(self, output, ln):
        # (# of sentences in the batch, 2*hidden) to (batch, max_sentence, 2*hidden)
        prev_idx = 0
        reorder_output = []
        for i in ln:
            sent_emb = output[prev_idx:prev_idx+i]
            sent_emb = F.pad(sent_emb, (0, 0, 0, self.max_sent-len(sent_emb)))
            reorder_output.append(sent_emb.unsqueeze(0))
            prev_idx += i

        return torch.cat(reorder_output)

    def forward(self, input, ln, ls, ent_embs, lk):
        # cat all sentences in the batch
        input, ls = self._reorder_input(input, ln, ls)

        # (# of sentences in the batch, max_length, emb_size)
        emb_w = self.embedding(input)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls, batch_first=True, enforce_sorted=False)
        sent_embs = self.word(packed_sents, emb_w.size(1))

        # recover sentence embs to batch
        sent_embs = self._reorder_word_output(sent_embs, ln)

        ## mask
        idxes = torch.arange(0, ent_embs.size(1), out=ent_embs.data.new(ent_embs.size(1))).unsqueeze(1)
        mask = (idxes>=lk.unsqueeze(0).to(idxes.device)).t() # (batch, max_ent)

        # news, entity, entity attention, get weighted ent_embed
        # Q sent: (batch, max_sent, 2*hidden)
        # V, K entity:(batch, max_ent, 2*hidden)
        ent_embs, ent_attn = self.NEE_attn(sent_embs.transpose(0, 1), ent_embs.transpose(0, 1), ent_embs.transpose(0, 1), key_padding_mask=mask)
        ent_embs = self.ent_lin(ent_embs) 
        ent_embs = self.relu(ent_embs)

        # cat weighted ent_embed to sent_embs
        sent_embs = torch.cat((sent_embs, ent_embs.transpose(0, 1)), dim=2) # (batch, max_sent, 3*hidden)

        packed_news = torch.nn.utils.rnn.pack_padded_sequence(sent_embs, ln, batch_first=True, enforce_sorted=False)
        content_vec = self.sent(packed_news, sent_embs.size(1))

        return content_vec, ent_attn # (batch, hid_size*2)


# comment hierarchical attention network
class CHAN(nn.Module):

    def __init__(self, word2vec, emb_size=100, hid_size=100, dropout=0.3):
        super(CHAN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec.vectors))
        self.word = AttentionalBiRNN(emb_size, hid_size, dropout=dropout)
        self.post = AttentionalBiRNN(hid_size*2, hid_size, dropout=dropout)
        self.subevent = AttentionalBiRNN(hid_size*2+hid_size//2, hid_size, dropout=dropout)
        
        self.SEE_attn = nn.MultiheadAttention(hid_size*2, 4)
        self.ent_lin = nn.Linear(hid_size*2, hid_size//2)
        self.relu = nn.ReLU()

    def _reorder_input(self, input, le, lsb, lc):
        # (batch, M, max_comment, max_length) to (# of comments in the batch, max_length)
        reorder_input = [sb[:lsb[i][j]] for i, event in enumerate(input) for j, sb in enumerate(event[:le[i]])]
        reorder_input = torch.cat(reorder_input, axis=0) if len(reorder_input) > 0 else torch.tensor([])

        reorder_lc = [k for i, l in enumerate(le) for j, s in enumerate(lsb[i][:l]) for k in lc[i][j][:s]]

        return reorder_input, reorder_lc

    def _reorder_word_output(self, output, le, lsb):
        # (# of comments in the batch, 2*hidden) to (# of subevents in the batch, max_comment, 2*hidden)
        reorder_lsb = [j for i, l in enumerate(le) for j in lsb[i][:l]]

        prev_idx = 0
        max_cmt = torch.max(lsb)
        reorder_output = []
        for i in reorder_lsb:
            cmt_emb = output[prev_idx:prev_idx+i]
            cmt_emb = F.pad(cmt_emb, (0, 0, 0, max_cmt-len(cmt_emb)))
            reorder_output.append(cmt_emb.unsqueeze(0))
            prev_idx += i

        return torch.cat(reorder_output), reorder_lsb

    def _reorder_post_output(self, output, le):
        # (# of subevents in the batch, 2*hidden) to (batch, M, 2*hidden)
        prev_idx = 0
        max_len = torch.max(le)
        reorder_output = []
        for i in le:
            sb_emb = output[prev_idx:prev_idx+i]
            sb_emb = F.pad(sb_emb, (0, 0, 0, max_len-len(sb_emb)))
            reorder_output.append(sb_emb.unsqueeze(0))
            prev_idx += i

        return torch.cat(reorder_output)
 
    def forward(self, input, le, lsb, lc, ent_embs, lk):
        # cat all comments in the batch
        input, lc = self._reorder_input(input, le, lsb, lc)

        # (# of comments in the batch, max_length, emb_size)
        emb_w = self.embedding(input)

        packed_cmts = torch.nn.utils.rnn.pack_padded_sequence(emb_w, lc, batch_first=True, enforce_sorted=False)
        post_embs = self.word(packed_cmts, emb_w.size(1))

        post_embs, lsb = self._reorder_word_output(post_embs, le, lsb)
        
        packed_sb = torch.nn.utils.rnn.pack_padded_sequence(post_embs, lsb, batch_first=True, enforce_sorted=False)
        sb_embs = self.post(packed_sb, post_embs.size(1))

        sb_embs = self._reorder_post_output(sb_embs, le)

        # mask
        idxes = torch.arange(0, ent_embs.size(1), out=ent_embs.data.new(ent_embs.size(1))).unsqueeze(1)
        mask = (idxes>=lk.unsqueeze(0).to(idxes.device)).t() # (batch, max_ent)

        # subevent, entity, entity attention, get weighted ent_embed
        # Q sb: (batch, M, 2*hidden)
        # V, K entity:(batch, max_ent, 2*hidden)
        ent_embs, ent_attn = self.SEE_attn(sb_embs.transpose(0, 1), ent_embs.transpose(0, 1), ent_embs.transpose(0, 1), key_padding_mask=mask)
        ent_embs = self.ent_lin(ent_embs) 
        ent_embs = self.relu(ent_embs)

        # cat weighted ent_embed to sb_embeds
        sb_embs = torch.cat((sb_embs, ent_embs.transpose(0, 1)), dim=2) # (batch, M, 3*hidden)

        packed_news = torch.nn.utils.rnn.pack_padded_sequence(sb_embs, le, batch_first=True, enforce_sorted=False)
        comment_vec = self.subevent(packed_news, sb_embs.size(1))

        return comment_vec, ent_attn # (batch, hid_size*2)


class KAHAN(nn.Module):

    def __init__(self, num_class, word2vec_cnt, word2vec_cmt, emb_size=100, hid_size=100, max_sent=50, dropout=0.3):
        super(KAHAN, self).__init__()

        self.news = NHAN(word2vec_cnt, emb_size, hid_size, max_sent, dropout)
        self.comment = CHAN(word2vec_cmt, emb_size, hid_size, dropout)
        self.lin_cat = nn.Linear(hid_size*4, hid_size*2)
        self.lin_out = nn.Linear(hid_size*2, num_class)
        self.relu = nn.ReLU()

    def attn_map(self, cnt_input, cmt_input, ent_input):
        # (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk)
        content_vec, n_ent_attn = self.news(*cnt_input, *ent_input)
        comment_vec, c_ent_attn = self.comment(*cmt_input, *ent_input) if torch.count_nonzero(cmt_input[-1]) > 0 else (torch.tensor([]), torch.tensor([]))

        out = torch.cat((content_vec, comment_vec), dim=1)

        out = self.lin_cat(out)
        out = self.relu(out)
        out = self.lin_out(out)

        return out, n_ent_attn, c_ent_attn

    def forward(self, cnt_input, cmt_input, ent_input):
        # (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk)
        content_vec,_ = self.news(*cnt_input, *ent_input)
        comment_vec,_ = self.comment(*cmt_input, *ent_input) if torch.count_nonzero(cmt_input[-1]) > 0 else (torch.ones(cmt_input[0].size(0), 200), torch.tensor([]))

        out = torch.cat((content_vec, comment_vec), dim=1)
        out = self.lin_cat(out)
        out = self.relu(out)
        out = self.lin_out(out)

        return out 


# model specific train function
def train(input_tensor, target_tensor, model, optimizer, criterion, device):
    (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk) = input_tensor
    cnt = cnt.to(device)
    cmt = cmt.to(device)
    ent = ent.to(device)
    target_tensor = target_tensor.to(device)

    model.train()
    optimizer.zero_grad()
    
    output = model((cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk))

    loss = criterion(output, target_tensor)

    correct = torch.sum(torch.eq(torch.argmax(output, -1), target_tensor)).item()

    loss.backward()
    optimizer.step()

    return loss.item(), correct

# model specific evaluation function
def evaluate(model, testset, device, batch_size=32):
    testloader = data.DataLoader(testset, batch_size)
    
    total = len(testset)
    correct = 0
    loss_total = 0
    criterion = nn.CrossEntropyLoss()
    predicts = []
    targets = []

    model.eval()
    with torch.no_grad():    
        for input_tensor, target_tensor in testloader:
            (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk) = input_tensor
            cnt = cnt.to(device)
            cmt = cmt.to(device)
            ent = ent.to(device)
            target_tensor = target_tensor.to(device)

            output = model((cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk))

            loss = criterion(output, target_tensor)
            loss_total += loss.item()*len(input_tensor)

            predicts.extend(torch.argmax(output, -1).tolist())
            targets.extend(target_tensor.tolist())
        
            correct += torch.sum(torch.eq(torch.argmax(output, -1), target_tensor)).item()

    return loss_total/total, correct/total, predicts, targets