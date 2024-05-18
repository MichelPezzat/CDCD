import torch
import torch.nn as nn
from synthesis.utils.misc import instantiate_from_config
from torch.cuda.amp import autocast
#import lib.losses.losses_utils as losses_utils
import math
import numpy as np
import torch.autograd.profiler as profiler
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GenericAux():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.cross_ent = nn.CrossEntropyLoss()



    def calc_loss(self, input,
                        return_loss=False,
                        return_logits=True,
            return_att_weight=False,
            is_train=True,
            **kwargs):
        
        sample_music = input['content_token'].type_as(input['content_token'])
 
        if self.condition_emb is not None:
            with autocast(enabled=False):
                with torch.no_grad():
                    # print("check condition_genre size:", input['condition_genre'].size())
                    cond_emb = self.condition_emb(input['condition_genre']) # B*360*219
                cond_emb = cond_emb.float()
        else: # share condition embeding with content
            if input.get('condition_e') == None:
                cond_emb = None
            else:
                cond_emb = input['condition_embed_token'].float()


       
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape
        device = self.transformer.device

        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = self.transformer.transition(ts) # (B, S, S)

        rate = self.transformer.rate(ts) # (B, S, S)


        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.flatten().long(),
            :
        ] # (B*D, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            x_t.long().flatten(),
            :
        ] # (B*D, S)
        rate_vals_square[
            torch.arange(B*D, device=device),
            x_t.long().flatten()
        ] = 0.0 # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, D, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, D)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample() # (B,) taking values in [0, D)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device),
            square_dims,
            :
        ] # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample() # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples
        # x_tilde (B, D)


        # ---------- First term of ELBO (regularization) ---------------


        if self.one_forward_pass:
            x_logits = self.transformer(sample_music,cond_emb ,ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
            reg_x = x_tilde
        else:
            x_logits = self.transformer(x_t, ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
            reg_x = x_t

        # For (B, D, S, S) first S is x_0 second S is x'

        mask_reg = torch.ones((B,D,S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            reg_x.long().flatten()
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)
        
        qt0_denom_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1,2) # (B, D, S)

        reg_term = torch.sum(
            (p0t_reg / qt0_denom_reg) * reg_tmp,
            dim=(1,2)
        )



        # ----- second term of continuous ELBO (signal term) ------------

        
        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            p0t_sig = F.softmax(self.transformer(x_tilde, ts), dim=2) # (B, D, S)

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        outer_qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.long().flatten(),
            x_tilde.long().flatten()
        ] + self.ratio_eps # (B, D)



        qt0_numer_sig = qt0.view(B, S, S) # first S is x_0, second S is x


        qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_tilde.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        ) # (B, D, S)


        x_tilde_mask = torch.ones((B,D,S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            x_tilde.long().flatten()
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B,D,S)

        outer_sum_sig = torch.sum(
            x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B,D,1)) * inner_log_sig,
            dim=(1,2)
        )

        # now getting the 2nd term normalization

        rate_row_sums = - rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B)
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(D),
            x_tilde.long().flatten()
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp # (B,D)
        Z_addition = rate_row_sums

        Z_sig_norm = base_Z.view(B, 1, 1) - \
            Z_subtraction.view(B, D, 1) + \
            Z_addition.view(B, 1, S)

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, D, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        qt0_sig_norm_denom = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.long().flatten(),
            x_tilde.long().flatten()
        ].view(B, D) + self.ratio_eps



        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B,D,1)),
            dim=(1,2)
        )

        sig_mean = torch.mean(- outer_sum_sig/sig_norm)

        reg_mean = torch.mean(reg_term)


        writer.add_scalar('sig', sig_mean.detach(), state['n_iter'])
        writer.add_scalar('reg', reg_mean.detach(), state['n_iter'])


        neg_elbo = sig_mean + reg_mean



        perm_x_logits = torch.permute(x_logits, (0,2,1))

        nll = self.cross_ent(perm_x_logits, minibatch.long())

        return neg_elbo + self.nll_weight * nll



class ConditionalAux(nn.Module):
    def __init__(self, 
        *,
        content_emb_config=None,
        condition_emb_config=None,
        transformer_config=None,
        ratio_eps=0,
        nll_weigth=0,
        min_time=0.01,
        one_forward_pass=False,
        condition_dim=32,
        ):
        
        super().__init__()


        if condition_emb_config is None:
            self.condition_emb = None
        else:
            self.condition_emb = instantiate_from_config(condition_emb_config)           

        transformer_config['params']['content_emb_config'] = content_emb_config
        self.transformer = instantiate_from_config(transformer_config)       

        self.num_classes = self.transformer.content_emb.num_embed
        self.shape = transformer_config['params']['content_seq_len']
        self.ratio_eps = ratio_eps
        self.nll_weigth = nll_weigth
        self.min_time = min_time
        self.one_forward_pass = one_forward_pass
        self.condition_dim = condition_dim
        self.lin = nn.Linear(256, 1024)
        self.cross_ent = nn.CrossEntropyLoss()


    #@property
    def device(self):
        return self.transformer.to_logits[-1].weight.device

    def calc_loss(self, input,
                        return_loss=False,
                        return_logits=True,
            return_att_weight=False,
            is_train=True,
            **kwargs):
     
        minibatch = input['content_token'].type_as(input['content_token'])
          
        S = self.num_classes
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape
        device = minibatch.device

        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = self.transformer.transition(ts, device) # (B, S, S)

        rate = self.transformer.rate(ts, device) # (B, S, S)

        conditioner = minibatch[:, 0:self.condition_dim]
        data = minibatch[:, self.condition_dim:]
        d = data.shape[1]

        if self.condition_emb is not None:
            with autocast(enabled=False):
                with torch.no_grad():
                    # print("check condition_genre size:", input['condition_genre'].size())
                    cond_emb = self.condition_emb(input['condition_genre']) # B*360*219
                cond_emb = cond_emb.float()
        else: # share condition embeding with content
            if input.get('condition_e') == None:
                cond_emb = None
            else:
                cond_emb = input['condition_embed_token'].float()
        cond_emb = self.lin(cond_emb)


        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.flatten().long(),
            :
        ] # (B*d, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, d)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(d),
            x_t.long().flatten(),
            :
        ] # (B*d, S)
        rate_vals_square[
            torch.arange(B*d, device=device),
            x_t.long().flatten()
        ] = 0.0 # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, d, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, d)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample() # (B,) taking values in [0, d)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device),
            square_dims,
            :
        ] # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample() # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples
        # x_tilde (B, d)


        # ---------- First term of ELBO (regularization) ---------------


        if self.one_forward_pass:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits = self.transformer(model_input,cond_emb ,ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits[:, self.condition_dim:, :], dim=2) # (B, d, S)
            reg_x = x_tilde
        else:
            model_input = torch.concat((conditioner, x_t), dim=1)
            x_logits = self.transformer(model_input, cond_emb,ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits[:, self.condition_dim:, :], dim=2) # (B, d, S)
            reg_x = x_t

        # For (B, d, S, S) first S is x_0 second S is x'

        mask_reg = torch.ones((B,d,S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            reg_x.long().flatten()
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)
        
        qt0_denom_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            reg_x.long().flatten()
        ].view(B, d, S) + self.ratio_eps

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            reg_x.long().flatten()
        ].view(B, d, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1,2) # (B, d, S)

        reg_term = torch.sum(
            (p0t_reg / qt0_denom_reg) * reg_tmp,
            dim=(1,2)
        )



        # ----- second term of continuous ELBO (signal term) ------------

        
        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits = self.transformer(model_input, cond_emb, ts) # (B, d, S)
            p0t_sig = F.softmax(x_logits[:, self.condition_dim:, :], dim=2) # (B, d, S)

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d*S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*d)
        ].view(B, d, S)

        outer_qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.long().flatten(),
            x_tilde.long().flatten()
        ] + self.ratio_eps # (B, d)



        qt0_numer_sig = qt0.view(B, S, S) # first S is x_0, second S is x


        qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            x_tilde.long().flatten()
        ].view(B, d, S) + self.ratio_eps

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        ) # (B, d, S)


        x_tilde_mask = torch.ones((B,d,S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            x_tilde.long().flatten()
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(d*S),
            torch.arange(S, device=device).repeat(B*d),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B,d,S)

        outer_sum_sig = torch.sum(
            x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B,d,1)) * inner_log_sig,
            dim=(1,2)
        )

        # now getting the 2nd term normalization

        rate_row_sums = - rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B)
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(d),
            x_tilde.long().flatten()
        ].view(B, d)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp # (B,d)
        Z_addition = rate_row_sums

        Z_sig_norm = base_Z.view(B, 1, 1) - \
            Z_subtraction.view(B, d, 1) + \
            Z_addition.view(B, 1, S)

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(d*S),
            torch.arange(S, device=device).repeat(B*d),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, d, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(d*S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*d)
        ].view(B, d, S)

        qt0_sig_norm_denom = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.long().flatten(),
            x_tilde.long().flatten()
        ].view(B, d) + self.ratio_eps



        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B,d,1)),
            dim=(1,2)
        )

        sig_mean = torch.mean(- outer_sum_sig/sig_norm)
        reg_mean = torch.mean(reg_term)

        writer.add_scalar('sig', sig_mean.detach(), state['n_iter'])
        writer.add_scalar('reg', reg_mean.detach(), state['n_iter'])

        out = {}

        neg_elbo = sig_mean + reg_mean
        out['neg_elbo'] = neg_elbo

        if return_logits:
             out['logits'] = x_logits

        perm_x_logits = torch.permute(x_logits, (0,2,1))

        nll = self.cross_ent(perm_x_logits, data.long())
        out['nll'] = nll

        loss = neg_elbo + self.nll_weight * nll

        if return_loss:
             out['loss'] = loss


        return out

    def sample(
            self,
            #condition_motion,
            #condition_video,
            condition_genre,
            condition_mask,
            condition_embed,
            content_token = None,
            num_steps=1,
            min_t=0.0,
            eps_ratio=0.0,
            reject_multiple_jumps=False,
            initial_dist=None,
            return_att_weight = False,
            return_logits = False,
            content_logits = None,
            print_log = True,
            num_intermediates=0,
            **kwargs):
        input = {'condition_motion': condition_motion,
                'condition_video': condition_video,
                'condition_genre': condition_genre,
                'content_token': content_token, 
                'condition_mask': condition_mask,
                'condition_embed_token': condition_embed,
                'content_logits': content_logits,
                }

        #if input['condition_motion'] != None:
         #   N = input['condition_motion'].shape[0]
        #else:
        N = kwargs['batch_size']
    
                
        #device = self.log_at.device
        device = input['condition_genre'].device
        start_step = int(self.num_timesteps * filter_ratio)

        # get cont_emb and cond_emb
        if content_token != None:
            conditioner = input['content_token'].type_as(input['content_token'])[:, 0:self.condition_dim]
        else:
            start_step = 0

        if self.condition_emb is not None:  # do this
            with torch.no_grad():
                cond_emb = self.condition_emb(input['condition_genre']) # B x Ld x D   #256*1024
            cond_emb = cond_emb.float()
        else: # share condition embeding with content
            if input.get('condition_embed_token', None) != None:
                cond_emb = input['condition_embed_token'].float()
            else:
                cond_emb = None
        #cond_motion = input['condition_motion']
        #cond_video = input['condition_video']
        # print("Check motion, video and genre condition embedding:", cond_motion.size(), cond_video.size(), cond_emb.size())
        #cond_emb = torch.cat((cond_motion, cond_video, cond_emb), 2)
        cond_emb = self.lin(cond_emb)


        t = 1.0
        total_D = self.shape
        sample_D = total_D - self.condition_dim
        S = self.num_classes

        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        #device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                qt0 = self.transfomer.transition(t * torch.ones((N,), device=device)) # (N, S, S)
                rate = self.transformer.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                model_input = torch.concat((conditioner, x), dim=1)
                p0t = F.softmax(self.transformer(model_input, cond_emb,t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                p0t = p0t[:, self.condition_dim:, :]


                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())



                qt0_denom = qt0[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N,sample_D,S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0 # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N, sample_D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(sample_D),
                    torch.arange(sample_D, device=device).repeat(N),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1,1,S) - x.view(N,sample_D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()

                if reject_multiple_jumps:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                    adj_diffs = masked_jump_nums * diffs
                else:
                    adj_diffs = jump_nums * diffs


                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(self.transfomer(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist
