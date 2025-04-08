import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from . import SASRecP, BERT4Rec
import random
import numpy as np
import math
import torch
import torch.nn.functional as F

num_train_timesteps = 1000



class Scheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = alphas_cumprod.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
        alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        
        self.betas = betas

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps=None
    ):
        bsz = original_samples.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                low=0, 
                high=self.num_train_timesteps, 
                size=(bsz,),
                device=original_samples.device
            )
        
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples, timesteps


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimDiff(SequentialRecommender):

    def __init__(self, config, dataset):
        super(SimDiff, self).__init__(config, dataset)

        # load parameters info
        self.initializer_range = config['initializer_range']

        self.mask_ratio = config['mask_ratio']
        self.generate_method = config['generate_method']

        self.mask_token = self.n_items
        self.hidden_size = config['hidden_size']  # same as embedding_size

        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        self.con_loss_fct = nn.CrossEntropyLoss()
        self.con_sim = Similarity(temp=0.05)

        self.mask_strategy = config['mask_strategy']

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.n_embedding = config['n_embedding']
        self.n_sampling = config['n_sampling']
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1

        self.encoder = SASRecP(config, dataset, self.item_embedding).to(config['device'])

        self.generator = BERT4Rec(config, dataset, self.item_embedding).to(config['device'])
        self.generator.trm_encoder = self.encoder.trm_encoder
        
        self.encode_loss_weight = config['encoder_loss_weight']
        self.con_loss_weight = config['contrastive_loss_weight']
        self.generate_loss_weight = config['generate_loss_weight']
        
        self.scheduler = Scheduler(num_train_timesteps)

        self.time_embed = nn.Embedding(num_train_timesteps, config['hidden_size'])
        
        self.apply(self._init_weights)
        self.encoder.apply(self._init_weights)

        self.recall, self.recall_n = 0, 0

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reset_parameters(self):
        self.apply(self._init_weights)
        self.encoder.apply(self._init_weights)
        self.generator.apply(self._init_weights)
        if hasattr(self, 'time_embed'):
            self.time_embed.apply(self._init_weights)
        if hasattr(self.item_embedding, 'reset_parameters'):
            self.item_embedding.reset_parameters()


    def forward(self, item_seq, item_seq_len):
        emb_weight = self.item_embedding.weight[:self.n_items].T
        
        masked_indices = None
        
        seq_emb = self.item_embedding(item_seq)
        
        sims = torch.matmul(seq_emb, emb_weight)
        
        batch_size, seq_len, _ = seq_emb.shape

        batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, seq_len)
        seq_idx = torch.arange(seq_len)[None, :].expand(batch_size, seq_len)

        sims[batch_idx, seq_idx, item_seq] = torch.finfo(emb_weight.dtype).min
        _, top_n_indices = torch.topk(sims, k=self.n_embedding, dim=-1)  # shape: [batch_size, seq_len, n]

        top_n_embeds = self.item_embedding(top_n_indices)  # [batch_size, seq_len, n, hidden_dim]

        noise = top_n_embeds.mean(dim=2)  # [batch_size, seq_len, hidden_dim]

        replaced_items, timesteps = self.scheduler.add_noise(seq_emb, noise)
        noise_embeds = replaced_items.view(batch_size, seq_len, -1)
        
        timesteps_full = timesteps.view(batch_size, 1)
        timestep_embeddings = self.time_embed(timesteps_full)
        logits, generate_loss, seq_output = self.generator.predictSeq_diffusion(item_seq, noise_embeds + timestep_embeddings, None)
        
        
        probs = F.softmax(logits, dim=-1)
        
        topk_probs, topk = torch.topk(probs, k=self.n_sampling, dim=-1)  # shape: [batch_size, seq_len, 2]
        
        max_probs, _ = torch.max(probs, dim=-1)  # shape: [batch_size, seq_len]
        max_probs[item_seq == 0] = 0
        _, topk_indices = torch.topk(max_probs, k=self.mask_item_length, dim=1)  # shape: [batch_size, k]

        masked_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=logits.device)
        masked_indices.scatter_(1, topk_indices, True)
        
        logits = torch.rand_like(logits)
        _, topk = torch.topk(logits, k=self.n_sampling, dim=-1)
        
        
        pos_index = topk[..., 0]   # shape: [batch, seq_len]
        neg_index = topk[..., -1]   # shape: [batch, seq_len]
        
        pos_seqs = item_seq.clone()
        pos_seqs[masked_indices] = pos_index[masked_indices]
        
        neg_seqs = item_seq.clone()
        neg_seqs[masked_indices] = neg_index[masked_indices]
        
        
        
        encode_output = self.encoder.forward(item_seq, item_seq_len)
        
        return encode_output, generate_loss, pos_seqs, neg_seqs


    def calculate_con_loss(self, seq_output, seq_output_1):
        logits_0 = seq_output[:, -1:, :].mean(dim=1).unsqueeze(1)
        logits_1 = seq_output_1[:, -1:, :].mean(dim=1).unsqueeze(0)
        cos_sim = self.con_sim(logits_0, logits_1).view(-1, seq_output_1.size(0))
        labels = torch.arange(logits_0.size(0)).long().to(self.device)
        con_loss = self.con_loss_fct(cos_sim, labels)

        return con_loss


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, generate_loss, pos_seqs, neg_seqs = self.forward(item_seq, item_seq_len)

        pos_items = interaction[self.POS_ITEM_ID]

        item_label = item_seq[:, 1:]
        pad = pos_items.unsqueeze(-1)
        item_labeln = torch.cat((item_label, pad), dim=-1).long().to(self.device)
        seq_emb = seq_output.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]
        test_item_emb = self.item_embedding.weight[:-1,:]
        logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
        pos_ids_l = torch.squeeze(item_labeln.view(-1))
        encode_loss = self.loss_fct(logits, pos_ids_l)

        con_loss = torch.tensor(0)
        if self.con_loss_weight != 0:
            # Used to calculate contrastive loss
            seq_output_1 = self.encoder.forward(pos_seqs, item_seq_len)
            seq_output_2 = self.encoder.forward(neg_seqs, item_seq_len)
            
            con_loss = self.calculate_con_loss(seq_output, torch.cat([seq_output_1, seq_output_2], dim=0))
        return self.encode_loss_weight * encode_loss, self.con_loss_weight * con_loss, self.generate_loss_weight * generate_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output_raw = self.encoder.forward(item_seq, item_seq_len)
        seq_output = seq_output_raw[:, -1, :].squeeze(1)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        # print(self.ITEM_SEQ)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output_raw = self.encoder.forward(item_seq, item_seq_len)
        seq_output = seq_output_raw[:, -1, :].squeeze(1)

        test_items_emb = self.item_embedding.weight[:-1, :]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        return scores, seq_output



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
