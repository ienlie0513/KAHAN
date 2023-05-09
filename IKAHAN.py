import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

print('torch version: ', torch.__version__)

import numpy as np

def length_to_cpu(length):
    # if not on cpu move it
    if isinstance(length, torch.Tensor):
        if length.device != torch.device('cpu'):
            length = length.cpu()
        length = length.numpy()
    return length

class MaxPooling():
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, x):
        return F.max_pool1d(x.unsqueeze(0), self.kernel_size).squeeze(0)

class AvgPooling():
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, x):
        return F.avg_pool1d(x.unsqueeze(0), self.kernel_size).squeeze(0)

class DeepFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super(DeepFC, self).__init__()

        self.deepfc = nn.Sequential()
        self.deepfc.add_module('input', nn.Linear(input_size, hidden_sizes[0]))
        self.deepfc.add_module('batchnorm_0', nn.BatchNorm1d(hidden_sizes[0]))
        self.deepfc.add_module('relu_0', nn.ReLU())
        self.deepfc.add_module('dropout_0', nn.Dropout(dropout))
        for i in range(len(hidden_sizes) - 1):
            self.deepfc.add_module('hidden_{}'.format(i), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.deepfc.add_module('batchnorm_{}'.format(i+1), nn.BatchNorm1d(hidden_sizes[i+1]))
            self.deepfc.add_module('relu_{}'.format(i+1), nn.ReLU())
            self.deepfc.add_module('dropout_{}'.format(i+1), nn.Dropout(dropout))
        self.deepfc.add_module('output', nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = self.deepfc(x)
        return x
    
class ImageAttention(nn.Module):
    def __init__(self, embed_size=512, num_heads=4):
        super(ImageAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads=num_heads)

    def forward(self, img_input, ent_vec, clm_vec, lk):
        # mask
        idxes = torch.arange(0, ent_vec.size(1), out=ent_vec.data.new(ent_vec.size(1))).unsqueeze(1)
        mask = (idxes>=lk.unsqueeze(0).to(idxes.device)).t() # (batch, max_ent)

        # add dimension for multihead attention
        img_input = img_input.unsqueeze(1)

        attn_output, _ = self.attn(img_input.permute(1, 0, 2), ent_vec.float().permute(1, 0, 2), clm_vec.float().permute(1, 0, 2), key_padding_mask=mask)
        return attn_output.permute(1, 0, 2).squeeze(1)

class EmbedAttention(nn.Module):
    def __init__(self, att_size):
        super(EmbedAttention, self).__init__()
        self.attn = nn.Linear(att_size, 1)

    def forward(self, input, len_s, total_length):
        ##print('input: {} len_s: {} total_length: {}'.format(input.size(), len_s, total_length))
        att = self.attn(input).squeeze(-1)
        ##print('att: {}'.format(att.size()))
        out = self._masked_softmax(att, len_s, total_length).unsqueeze(-1)
        ##print('out: {}'.format(out.size()))

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
        ##print('packed_batch: {} total_length: {}'.format(packed_batch.data.size(), total_length))
        rnn_output, _ = self.rnn(packed_batch)
        ##print('rnn_output: {}'.format(rnn_output.data.size()))
        enc_output, len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True, total_length=length_to_cpu(total_length))
        ##print('enc_output: {} len_s: {}'.format(enc_output.data.size(), len_s))

        enc_output = self.dropout(enc_output)
        ##print('enc_output: {}'.format(enc_output.size()))
        emb_h = torch.tanh(self.lin(enc_output))
        ##print('emb_h: {}'.format(emb_h.size()))

        attended = self.emb_att(emb_h, len_s, total_length) * enc_output
        ##print('attended: {}'.format(attended.size()))
        
        return attended.sum(1)
    

# image hierachical attention network
class IHAN(nn.Module):
    def __init__(self, emb_size=100, hid_size=100, use_attention=True, dropout=0.3):
        super(IHAN, self).__init__()
        self.fine = AttentionalBiRNN(emb_size, hid_size, dropout=dropout) 
        self.use_attention = use_attention
        course_input_size = hid_size*2+hid_size//2 if use_attention else hid_size*2
        self.coarse = AttentionalBiRNN(course_input_size, hid_size, dropout=dropout)

        self.IME_attn = nn.MultiheadAttention(hid_size*2, 4)
        self.ent_lin = nn.Linear(hid_size*2, hid_size//2)
        self.relu = nn.ReLU()

    def generate_lengths(self, input):
        # input: (batch, coarse, fine, emb_size)
        # output: lc: (batch) lf: (batch, coarse) 
        lc = [input.shape[1] for _ in range(input.shape[0])]
        lf = [[input.shape[2] for _ in range(input.shape[1])] for _ in range(input.shape[0])]

        return np.asarray(lc), np.asarray(lf)

    def _reorder_input(self, input, lc, lf):
        # (batch, coarse, fine, emb_size) to (# of coarse in the batch, fine, emb_size)
        reorder_input = [imgs[:lc[i]] for i, imgs in enumerate(input)]
        reorder_input = torch.cat(reorder_input, axis=0)

        reorder_lf = [j for i, l in enumerate(lc) for j in lf[i][:l]]

        return reorder_input, reorder_lf

    def _reorder_fine_output(self, output, lc):
        # (# of sentences in the batch, 2*hidden) to (batch, max_sentence, 2*hidden)
        prev_idx = 0
        reorder_output = []
        max_coarse = max(lc) # TODO: might need changing if padding actually included in the data
        for i in lc:
            coarse_emb = output[prev_idx:prev_idx+i]
            coarse_emb = F.pad(coarse_emb, (0, 0, 0, max_coarse-len(coarse_emb)))
            reorder_output.append(coarse_emb.unsqueeze(0))
            prev_idx += i

        return torch.cat(reorder_output)
        # # (# of coarse in the batch, 2*hidden) to (batch, max_coarse, 2*hidden)
        # reordered_output = output.reshape(-1, lc, output.shape[1])

        # return reordered_output

    def forward(self, input, ent_embs, lk):
        lc, lf = self.generate_lengths(input)
        ##print('input: {} lf: {} lc: {}'.format(input.shape, lf.shape, lc.shape))
        input, len_f = self._reorder_input(input, lc, lf)
        ##print('input: {} len_f: {}'.format(input.size(), len_f))
        # fine graied level
        packed_fined = torch.nn.utils.rnn.pack_padded_sequence(input, len_f, batch_first=True, enforce_sorted=False)
        ##print('packed_fined: {} dtype: {}'.format(packed_fined.data.size(), packed_fined.data.dtype))
        fine_embs = self.fine(packed_fined, input.size(1))
        ##print('fine_embs: {}'.format(fine_embs.size()))
        # coarse grained level
        coarse_embs = self._reorder_fine_output(fine_embs, lc)
        ##print('coarse_embs: {}'.format(coarse_embs.size()))

        if self.use_attention:
            ## mask
            idxes = torch.arange(0, ent_embs.size(1), out=ent_embs.data.new(ent_embs.size(1))).unsqueeze(1)
            mask = (idxes>=lk.unsqueeze(0).to(idxes.device)).t() # (batch, max_ent)

            # image, entity, entity attention, get weighted ent_embed
            # Q sent: (batch, max_coarse, 2*hidden)
            # V, K entity:(batch, max_ent, 2*hidden)
            ent_embs, ent_attn = self.IME_attn(coarse_embs.transpose(0, 1), ent_embs.transpose(0, 1), ent_embs.transpose(0, 1), key_padding_mask=mask)
            ent_embs = self.ent_lin(ent_embs) 
            ent_embs = self.relu(ent_embs)

            # cat weighted ent_embed to sent_embs
            coarse_embs = torch.cat((coarse_embs, ent_embs.transpose(0, 1)), dim=2) # (batch, max_sent, 3*hidden)
            ##print('sent_embs: {} '.format(sent_embs.size()))

        packed_coarse = torch.nn.utils.rnn.pack_padded_sequence(coarse_embs, lc, batch_first=True, enforce_sorted=False)
        ##print('packed_coarse: {}'.format(packed_coarse.data.size()))
        image_vec = self.coarse(packed_coarse, coarse_embs.size(1))
        ##print('image_vec: {}'.format(image_vec.size()))

        return image_vec


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
        ##print('input (0): {} ln: {} ls: {} '.format(input.size(), ln.shape, ls.shape))
        input, ls = self._reorder_input(input, ln, ls)
        ##print('input (1): {} ls: {} ls_len: {} '.format(input.size(), ls, len(ls)))

        # (# of sentences in the batch, max_length, emb_size)
        emb_w = self.embedding(input) 
        ##print('emb_w: {} '.format(emb_w.size()))
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls, batch_first=True, enforce_sorted=False)
        ##print('packed_sents: {} dtype {} '.format(packed_sents.data.size(), packed_sents.data.dtype))
        sent_embs = self.word(packed_sents, emb_w.size(1))
        ##print('sent_embs (0): {} '.format(sent_embs.size()))

        # recover sentence embs to batch
        sent_embs = self._reorder_word_output(sent_embs, ln)
        ##print('sent_embs (1): {} '.format(sent_embs.size()))

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
        ##print('sent_embs: {} '.format(sent_embs.size()))

        ##print('sent_embs: {} num sentences (ln) {} '.format(sent_embs.size(), ln))
        packed_news = torch.nn.utils.rnn.pack_padded_sequence(sent_embs, length_to_cpu(ln), batch_first=True, enforce_sorted=False)
        ##print('packed_news: {} '.format(packed_news.data.size()))
        content_vec = self.sent(packed_news, sent_embs.size(1))
        ##print('content_vec: {} '.format(content_vec.size()))

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
    
    def _filter_empty_cmts(self, mask, input, le, lsb, lc, ent_embs, lk):
        idx = mask.nonzero(as_tuple=True)[0]

        # filter the input comments and lengths using the index array
        input_filtered = input[idx]
        le_filtered = le[idx]
        lsb_filtered = lsb[idx]
        lc_filtered = lc[idx]
        ent_embs_filtered = ent_embs[idx]
        lk_filtered = lk[idx]

        return input_filtered, le_filtered, lsb_filtered, lc_filtered, ent_embs_filtered, lk_filtered

 
    def forward(self, input, le, lsb, lc, ent_embs, lk):
        # non empty comments mask
        non_empty_mask = le.gt(0)

        # return filtered arguments so we omitt processing of empty comments
        input_flt, le_flt, lsb_flt, lc_flt, ent_embs_flt, lk_flt = self._filter_empty_cmts(non_empty_mask, input, le, lsb, lc, ent_embs, lk)
        input_flt, lc_flt = self._reorder_input(input_flt, le_flt, lsb_flt, lc_flt)

        # (# of comments in the batch, max_length, emb_size)
        emb_w = self.embedding(input_flt.long())
        packed_cmts = torch.nn.utils.rnn.pack_padded_sequence(emb_w, lc_flt, batch_first=True, enforce_sorted=False)
        post_embs = self.word(packed_cmts, emb_w.size(1))

        post_embs, lsb_flt = self._reorder_word_output(post_embs, le_flt, lsb_flt)

        packed_sb = torch.nn.utils.rnn.pack_padded_sequence(post_embs, lsb_flt, batch_first=True, enforce_sorted=False)
        sb_embs = self.post(packed_sb, post_embs.size(1))
        sb_embs = self._reorder_post_output(sb_embs, le_flt)

        # mask
        idxes = torch.arange(0, ent_embs_flt.size(1), out=ent_embs_flt.data.new(ent_embs_flt.size(1))).unsqueeze(1)
        mask = (idxes>=lk_flt.unsqueeze(0).to(idxes.device)).t() # (batch, max_ent)

        # subevent, entity, entity attention, get weighted ent_embed
        # Q sb: (batch, M, 2*hidden)
        # V, K entity:(batch, max_ent, 2*hidden)
        ent_embs_flt, ent_attn_flt = self.SEE_attn(sb_embs.transpose(0, 1), ent_embs_flt.transpose(0, 1), ent_embs_flt.transpose(0, 1), key_padding_mask=mask)
        ent_embs_flt = self.ent_lin(ent_embs_flt) 
        ent_embs_flt = self.relu(ent_embs_flt)

        # cat weighted ent_embed to sb_embeds
        sb_embs = torch.cat((sb_embs, ent_embs_flt.transpose(0, 1)), dim=2) # (batch, M, 3*hidden)

        packed_news = torch.nn.utils.rnn.pack_padded_sequence(sb_embs, length_to_cpu(le_flt), batch_first=True, enforce_sorted=False)
        comment_vec_flt = self.subevent(packed_news, sb_embs.size(1))
        
        # reconstructing the original comment embeddings and attention by adding zero padding to make a 
        comment_vec = torch.zeros(input.size(0), comment_vec_flt.size(1)).to(comment_vec_flt.device)
        ent_attn = torch.zeros(input.size(0), ent_attn_flt.size(1), ent_attn_flt.size(2)).to(ent_attn_flt.device)

        comment_vec[non_empty_mask] = comment_vec_flt
        ent_attn[non_empty_mask] = ent_attn_flt

        return comment_vec, ent_attn # (batch, hid_size*2)

    
class DimentionalityReduction(nn.Module):
    def __init__(self, out_size, method, embed_size, kernel_size, hid_layers):
        super(DimentionalityReduction, self).__init__()

        if method == 'maxpooling':
            self.model = MaxPooling(kernel_size)
        elif method == 'avgpooling':
            self.model = AvgPooling(kernel_size)
        elif method == 'deepfc':
            self.model = DeepFC(embed_size, hid_layers, out_size)
        elif method == 'fc':
            self.model = nn.Linear(embed_size, out_size)
        
    def forward(self, embed):
        return self.model(embed)

class IKAHAN(nn.Module):

    def __init__(self, num_class, word2vec_cnt, word2vec_cmt, dimred_params, kahan, deep_classifier, fusion_method, device, ihan=False, clip=False, img_ent_att=False, clip_emb_size=512, emb_size=100, hid_size=100, max_sent=50, max_len=120, max_cmt=50, dropout=0.3):
        super(IKAHAN, self).__init__()

        self.device = device

        self.news = NHAN(word2vec_cnt, emb_size, hid_size, max_sent, dropout)
        self.comment = CHAN(word2vec_cmt, emb_size, hid_size, dropout)
        self.image = IHAN(emb_size, hid_size, img_ent_att, dropout) if ihan else DimentionalityReduction(hid_size*2, **dimred_params)
        self.img_att = ImageAttention(clip_emb_size, 4)

        self.word2vec_cmt = word2vec_cmt
        self.hid_size = hid_size
        self.max_cmt = max_cmt
        self.max_len = max_len

        self.kahan = kahan
        self.deep_classifier = deep_classifier

        self.fusion_method = fusion_method
        self.ihan = ihan
        self.clip = clip
        self.img_ent_att = img_ent_att

        if self.kahan:
            in_dim = hid_size*4
        else:
            if self.clip:
                in_dim = hid_size*4 + clip_emb_size
            else:
                in_dim = hid_size*6

        if self.deep_classifier:
            self.lin_out = DeepFC(in_dim, [hid_size*2, hid_size], num_class, dropout=dropout)
        else:
            self.lin_cat = nn.Linear(in_dim, hid_size*2)
            self.lin_out = nn.Linear(hid_size*2, num_class)
            self.relu = nn.ReLU()

    def attn_map(self, cnt_input, cmt_input, ent_input):
        # (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk)
        content_vec, n_ent_attn = self.news(*cnt_input, *ent_input)
        content_vec.to(self.device)

        # comment_vec, c_ent_attn = self.comment(*cmt_input, *ent_input) if not torch.equal(cmt_input[0], np.full((self.max_cmt, self.max_len), self.word2vec_cmt.key_to_index['_pad_'], dtype=int)) else (torch.ones(cmt_input[0].size(0), self.hid_size*2).to(self.device), torch.tensor([], device=self.device))
        comment_vec, c_ent_attn = self.comment(*cmt_input, *ent_input) if all(cmt_input[1] > 0) else (torch.ones(cmt_input[0].size(0), self.hid_size*2).to(self.device), torch.tensor([], device=self.device))
        comment_vec.to(self.device)

        out = torch.cat((content_vec, comment_vec), dim=1)
        out = self.lin_cat(out)
        out = self.relu(out)
        out = self.lin_out(out)

        return out, n_ent_attn, c_ent_attn

    def forward(self, cnt_input, cmt_input, ent_input, clip_ent_input, img_input):
        # (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk), (clip_ent, lk), img
        content_vec = self.news(*cnt_input, *ent_input)[0].to(self.device)
        comment_vec = self.comment(*cmt_input, *ent_input)[0].to(self.device)

        if self.clip:
            if self.img_ent_att:
                image_vec = self.img_att(img_input, *clip_ent_input).to(self.device)
            else:
                image_vec = img_input
        else:
            if self.ihan:
                image_vec = self.image(img_input, *ent_input).to(self.device)
            else:
                image_vec = self.image(img_input).to(self.device)

        if self.kahan:
            out = torch.cat((content_vec, comment_vec), dim=1)
        else:
            if self.clip:
                out = torch.cat((content_vec, comment_vec, image_vec), dim=1)
            else:
                if self.fusion_method == 'cat':
                    out = torch.cat((content_vec, comment_vec, image_vec), dim=1)
                elif self.fusion_method == 'elem_mult':
                    out = content_vec * comment_vec * image_vec
                elif self.fusion_method == 'avg':
                    out = (content_vec + comment_vec + image_vec) / 3

        if not self.deep_classifier:        
            out = self.lin_cat(out)
            out = self.relu(out)

        out = self.lin_out(out)

        return out

# model specific train function
def train(input_tensor, target_tensor, model, optimizer, criterion, device):
    (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk), (clip_ent, clip_clm, clip_lk), img = input_tensor

    cnt = cnt.to(device)
    cmt = cmt.to(device)
    ent = ent.to(device)
    clip_ent = clip_ent.to(device)
    clip_clm = clip_clm.to(device)
    img = img.to(device)
    target_tensor = target_tensor.to(device)

    model.train()
    optimizer.zero_grad()
    
    output = model((cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk), (clip_ent, clip_clm, clip_lk), img)

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
            (cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk), (clip_ent, clip_clm, clip_lk), img = input_tensor
            cnt = cnt.to(device)
            cmt = cmt.to(device)
            ent = ent.to(device)
            clip_ent = clip_ent.to(device)
            clip_clm = clip_clm.to(device)
            img = img.to(device)
            target_tensor = target_tensor.to(device)

            output = model((cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk), (clip_ent, clip_clm, clip_lk), img)

            loss = criterion(output, target_tensor)
            loss_total += loss.item()*len(input_tensor)

            predicts.extend(torch.argmax(output, -1).tolist())
            targets.extend(target_tensor.tolist())
        
            correct += torch.sum(torch.eq(torch.argmax(output, -1), target_tensor)).item()

    return loss_total/total, correct/total, predicts, targets