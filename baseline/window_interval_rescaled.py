import math
import types
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch import nn
from baseline.AKT import AKT, CosinePositionalEmbedding
from baseline.SAKT import SAKT
from baseline.folibikt import folibiKT
from baseline.cl4kt import CL4KT
from baseline.robustkt import Robustkt
from baseline.simplekt import simpleKT
from baseline.sparsekt import sparseKT

class CosineHistoryLengthEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        cpu_rng_state = torch.get_rng_state()
        self.position = CosinePositionalEmbedding(d_model=d_model, max_len=max_len + 1)
        torch.set_rng_state(cpu_rng_state)

    def forward(self, history_lengths):
        indices = history_lengths.clamp_max(self.position.weight.size(1) - 1)
        return self.position.weight[0][indices]

def _causal_concept_mask(dcur):
    concepts = dcur['cseqs'].long()
    shifted_concepts = dcur['shft_cseqs'].long()
    q_data = torch.cat((concepts[:, 0:1], shifted_concepts), dim=1)
    return q_data.ne(0)

class IntervalTargetMixin:

    def _init_interval_targets(self, d_model, interval_step=2, max_seq_len=200, index_dim=32, hidden_dim=128):
        if interval_step <= 0:
            raise ValueError('interval_step must be positive')
        self.interval_step = int(interval_step)
        self.interval_max_seq_len = int(max_seq_len)
        self.interval_start_embedding = nn.Embedding(max_seq_len + 1, index_dim)
        self.interval_end_embedding = nn.Embedding(max_seq_len + 1, index_dim)
        self.interval_history_embedding = CosineHistoryLengthEncoding(d_model=index_dim, max_len=max_seq_len)
        self.interval_state_norm = nn.LayerNorm(d_model)
        target_input_dim = d_model + 3 * index_dim
        self.interval_target_net = nn.Sequential(nn.Linear(target_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.interval_target_net[-1].weight)
        nn.init.zeros_(self.interval_target_net[-1].bias)
        self.last_matching_loss = None
        self.last_normalization_loss = None
        self.last_balance_loss = None
        self.last_interval_mass = None
        self.last_target_mass = None

    def _prepare_attention(self, valid):
        if self._attention is None:
            return (None, None, None)
        attention = self._attention
        if attention.dim() == 4:
            attention = attention.mean(dim=1)
        if attention.dim() != 3:
            raise RuntimeError(f'Unexpected attention shape: {tuple(attention.shape)}')
        _, seq_len, _ = attention.shape
        device = attention.device
        positions = torch.arange(seq_len, device=device)
        distance = positions[:, None] - positions[None, :]
        history_mask = distance.gt(0).unsqueeze(0)
        history_mask = history_mask & valid[:, None, :] & valid[:, :, None]
        attention = attention * history_mask.to(attention.dtype)
        return (attention, history_mask, distance)

    def _build_intervals(self, history_mask):
        _, seq_len, _ = history_mask.shape
        device = history_mask.device
        max_history = min(seq_len - 1, self.interval_max_seq_len)
        interval_starts = torch.arange(0, max_history, self.interval_step, device=device)
        nominal_ends = interval_starts + self.interval_step
        history_lengths = history_mask.sum(dim=-1).clamp_max(self.interval_max_seq_len).long()
        actual_ends = torch.minimum(nominal_ends.view(1, 1, -1), history_lengths.unsqueeze(-1))
        active = interval_starts.view(1, 1, -1).lt(history_lengths.unsqueeze(-1))
        return (interval_starts, nominal_ends, actual_ends, history_lengths, active)

    def _interval_attention_mass(self, attention, distance, interval_starts, nominal_ends):
        interval_masks = distance.unsqueeze(0).gt(interval_starts[:, None, None]) & distance.unsqueeze(0).le(nominal_ends[:, None, None])
        interval_mass = torch.einsum('bij,kij->bik', attention, interval_masks.to(attention.dtype))
        return interval_mass

    def _interval_target_mass(self, interval_starts, actual_ends, history_lengths):
        state = self.interval_state_norm(self._state)
        batch_size, seq_len, _ = state.shape
        num_intervals = interval_starts.numel()
        state_features = state.unsqueeze(2).expand(-1, -1, num_intervals, -1)
        start_features = self.interval_start_embedding(interval_starts).view(1, 1, num_intervals, -1).expand(batch_size, seq_len, -1, -1)
        end_features = self.interval_end_embedding(actual_ends)
        history_features = self.interval_history_embedding(history_lengths).unsqueeze(2).expand(-1, -1, num_intervals, -1)
        features = torch.cat((state_features, start_features, end_features, history_features), dim=-1)
        target_mass = torch.sigmoid(self.interval_target_net(features).squeeze(-1))
        return target_mass

    def interval_balance_loss(self, valid):
        if self._attention is None or self._state is None:
            return next(self.interval_target_net.parameters()).new_zeros(())
        attention, history_mask, distance = self._prepare_attention(valid)
        interval_starts, nominal_ends, actual_ends, history_lengths, active = self._build_intervals(history_mask)
        interval_mass = self._interval_attention_mass(attention, distance, interval_starts, nominal_ends)
        target_mass = self._interval_target_mass(interval_starts, actual_ends, history_lengths)
        valid_rows = valid & history_lengths.gt(0)
        active = active & valid_rows.unsqueeze(-1)
        if not active.any():
            return attention.new_zeros(())
        matching_loss = (interval_mass[active] - target_mass[active]).square().mean()
        target_sum = (target_mass * active.to(target_mass.dtype)).sum(dim=-1)
        normalization_loss = (target_sum[valid_rows] - 1.0).abs().mean()
        balance_loss = matching_loss + normalization_loss
        self.last_matching_loss = matching_loss.detach()
        self.last_normalization_loss = normalization_loss.detach()
        self.last_balance_loss = balance_loss.detach()
        self.last_interval_mass = interval_mass.detach()
        self.last_target_mass = target_mass.detach()
        return balance_loss

    def auxiliary(self, valid):
        return self.interval_balance_loss(valid)

class ExplicitIntervalRescaleMixin:

    def _init_rescaling(self, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0):
        self.rescale_eps = float(rescale_eps)
        # Fixed experiment coefficients for alpha * f_k / (w_k + eps) + beta.
        self.register_buffer('alpha', torch.tensor(float(rescale_alpha)), persistent=False)
        self.register_buffer('beta', torch.tensor(float(rescale_beta)), persistent=False)
        self.last_rescale_ratio = None
        self.last_linear_scale = None
        self.last_rescaled_row_sum_error = None

    def _interval_scale(self, valid):
        attention, history_mask, distance = self._prepare_attention(valid)
        if attention is None:
            return None
        starts, ends, actual_ends, history_lengths, active = self._build_intervals(history_mask)
        observed = self._interval_attention_mass(attention, distance, starts, ends)
        target = self._interval_target_mass(starts, actual_ends, history_lengths)
        valid_rows = valid & history_lengths.gt(0)
        active = active & valid_rows.unsqueeze(-1)
        interval_ratio = torch.ones_like(target)
        interval_ratio = torch.where(
            active,
            target / (observed + self.rescale_eps),
            interval_ratio,
        )
        linear_scale = self.alpha * interval_ratio + self.beta
        scale = attention.new_ones(attention.shape)
        for index in range(starts.numel()):
            interval = (distance.gt(starts[index]) & distance.le(ends[index])).unsqueeze(0)
            mask = interval & active[..., index].unsqueeze(-1)
            scale = torch.where(mask, linear_scale[..., index].unsqueeze(-1), scale)
        self.last_rescale_ratio = interval_ratio.detach()
        self.last_linear_scale = linear_scale.detach()
        return scale

    @contextmanager
    def _rescaled_dropout(self, dropout, scale):
        original = dropout.forward

        def forward(module, attention):
            if attention.dim() != 4:
                return original(attention)
            scaled = attention * scale.unsqueeze(1)
            denominator = scaled.sum(dim=-1, keepdim=True)
            scaled = torch.where(denominator.gt(self.rescale_eps), scaled / denominator.clamp_min(self.rescale_eps), scaled)
            self.last_rescaled_row_sum_error = scaled.sum(dim=-1).sub(1.0).abs().masked_select(denominator.squeeze(-1).gt(self.rescale_eps)).max().detach() if denominator.gt(self.rescale_eps).any() else scaled.new_zeros(())
            return F.dropout(scaled, p=module.p, training=module.training, inplace=module.inplace)
        dropout.forward = types.MethodType(forward, dropout)
        try:
            yield
        finally:
            dropout.forward = original

    def _set_auxiliary(self, valid):
        balance = self.interval_balance_loss(valid)
        if self.training and self.window_reg_weight > 0:
            return self.window_reg_weight * balance
        return balance.new_zeros(())

class AKTIntervalTargets(IntervalTargetMixin, AKT):

    def __init__(self, *args, window_reg_weight=0.01, interval_step=2, interval_max_seq_len=200, interval_index_dim=32, interval_hidden_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_reg_weight = float(window_reg_weight)
        self._attention = None
        self._state = None
        self._init_interval_targets(d_model=int(kwargs['d_model']), interval_step=interval_step, max_seq_len=interval_max_seq_len, index_dim=interval_index_dim, hidden_dim=interval_hidden_dim)
        self.model.register_forward_hook(self._capture_state)
        final_retriever = self.model.blocks_2[-1].masked_attn_head
        final_retriever.dropout.register_forward_pre_hook(self._capture_attention)

    def _capture_attention(self, module, args):
        self._attention = args[0]

    def _capture_state(self, module, inputs, output):
        self._state = output

    def forward(self, q_data, target, pid_data=None, qtest=False):
        self._attention = None
        self._state = None
        result = super().forward(q_data, target, pid_data=pid_data, qtest=qtest)
        if qtest:
            predictions, regularization, features = result
        else:
            predictions, regularization = result
        if self.training and self.window_reg_weight > 0:
            valid = q_data.ne(0)
            balance_loss = self.auxiliary(valid)
            regularization = regularization + self.window_reg_weight * balance_loss
        if qtest:
            return (predictions, regularization, features)
        return (predictions, regularization)

class SAKTIntervalTargets(IntervalTargetMixin, SAKT):

    def __init__(self, *args, window_reg_weight=0.01, interval_step=2, interval_max_seq_len=200, interval_index_dim=32, interval_hidden_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_reg_weight = float(window_reg_weight)
        self._attention = None
        self._state = None
        self.window_aux_loss = torch.tensor(0.0)
        self._init_interval_targets(d_model=int(kwargs['emb_size']), interval_step=interval_step, max_seq_len=interval_max_seq_len, index_dim=interval_index_dim, hidden_dim=interval_hidden_dim)
        self.blocks[-1].register_forward_hook(self._capture_state)
        self.blocks[-1].attn.register_forward_hook(self._capture_attention)

    def _capture_state(self, module, inputs, output):
        self._state = output

    def _capture_attention(self, module, inputs, output):
        self._attention = output[1]

    def forward(self, q, r, qry, qtest=False):
        self._attention = None
        self._state = None
        result = super().forward(q, r, qry, qtest=qtest)
        device = self._state.device if self._state is not None else q.device
        self.window_aux_loss = torch.zeros((), device=device)
        if self.training and self.window_reg_weight > 0:
            valid = qry.ne(0)
            balance_loss = self.auxiliary(valid)
            self.window_aux_loss = self.window_reg_weight * balance_loss
        return result

class FoLiBiIntervalTargets(IntervalTargetMixin, folibiKT):

    def __init__(self, *args, window_reg_weight=0.01, interval_step=2, interval_max_seq_len=200, interval_index_dim=32, interval_hidden_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_reg_weight = float(window_reg_weight)
        self._attention = None
        self._state = None
        self._init_interval_targets(d_model=int(kwargs['d_model']), interval_step=interval_step, max_seq_len=interval_max_seq_len, index_dim=interval_index_dim, hidden_dim=interval_hidden_dim)
        self.model.register_forward_hook(self._capture_state)
        final_retriever = self.model.blocks_2[-1].masked_attn_head
        final_retriever.dropout.register_forward_pre_hook(self._capture_attention)

    def _capture_attention(self, module, args):
        self._attention = args[0]

    def _capture_state(self, module, inputs, output):
        self._state = output

    def forward(self, q_data, target, pid_data=None, qtest=False):
        self._attention = None
        self._state = None
        result = super().forward(q_data, target, pid_data=pid_data, qtest=qtest)
        if qtest:
            predictions, regularization, features = result
        else:
            predictions, regularization = result
        if self.training and self.window_reg_weight > 0:
            valid = q_data.ne(0)
            balance_loss = self.auxiliary(valid)
            regularization = regularization + self.window_reg_weight * balance_loss
        if qtest:
            return (predictions, regularization, features)
        return (predictions, regularization)

class RobustKTIntervalTargets(IntervalTargetMixin, Robustkt):

    def __init__(self, *args, window_reg_weight=0.01, interval_step=2, interval_max_seq_len=200, interval_index_dim=32, interval_hidden_dim=128, **kwargs):
        self.window_reg_weight = float(window_reg_weight)
        super().__init__(*args, **kwargs)
        self._init_interval_targets(d_model=int(kwargs['d_model']), interval_step=interval_step, max_seq_len=interval_max_seq_len, index_dim=interval_index_dim, hidden_dim=interval_hidden_dim)
        self._attention = None
        self._state = None
        self.model.register_forward_hook(self._capture_state)
        final_retriever = self.model.blocks_2[-1].masked_attn_head
        final_retriever.dropout.register_forward_pre_hook(self._capture_attention)

    def _capture_state(self, module, inputs, output):
        self._state = output

    def _capture_attention(self, module, args):
        self._attention = args[0]

    def forward(self, q_data, target, pid_data=None, qtest=False):
        self._attention = None
        self._state = None
        result = super().forward(q_data, target, pid_data=pid_data, qtest=qtest)
        if qtest:
            predictions, regularization, features = result
        else:
            predictions, regularization = result
        if self.training and self.window_reg_weight > 0:
            valid = q_data.ne(0)
            balance_loss = self.auxiliary(valid)
            regularization = regularization + self.window_reg_weight * balance_loss
        if qtest:
            return (predictions, regularization, features)
        return (predictions, regularization)

class SimpleKTIntervalTargets(IntervalTargetMixin, simpleKT):

    def __init__(self, *args, window_reg_weight=0.01, interval_step=2, interval_max_seq_len=200, interval_index_dim=32, interval_hidden_dim=128, **kwargs):
        self.window_reg_weight = float(window_reg_weight)
        super().__init__(*args, **kwargs)
        self._init_interval_targets(d_model=int(kwargs['d_model']), interval_step=interval_step, max_seq_len=interval_max_seq_len, index_dim=interval_index_dim, hidden_dim=interval_hidden_dim)
        self._attention = None
        self._state = None
        self.window_aux_loss = None
        self.model.register_forward_hook(self._capture_state)
        final_retriever = self.model.blocks_2[-1].masked_attn_head
        final_retriever.dropout.register_forward_pre_hook(self._capture_attention)

    def _capture_state(self, module, inputs, output):
        self._state = output

    def _capture_attention(self, module, args):
        self._attention = args[0]

    def forward(self, dcur, qtest=False, train=False):
        self._attention = None
        self._state = None
        result = super().forward(dcur, qtest=qtest, train=train)
        if not train:
            return result
        predictions, y2, y3 = result
        self.window_aux_loss = next(self.interval_target_net.parameters()).new_zeros(())
        if self.training and self.window_reg_weight > 0:
            valid = _causal_concept_mask(dcur)
            balance_loss = self.auxiliary(valid)
            self.window_aux_loss = self.window_reg_weight * balance_loss
        return (predictions, y2, y3)

class SparseKTIntervalTargets(IntervalTargetMixin, sparseKT):

    def __init__(self, *args, window_reg_weight=0.01, interval_step=2, interval_max_seq_len=200, interval_index_dim=32, interval_hidden_dim=128, **kwargs):
        self.window_reg_weight = float(window_reg_weight)
        super().__init__(*args, **kwargs)
        self._init_interval_targets(d_model=int(kwargs['d_model']), interval_step=interval_step, max_seq_len=interval_max_seq_len, index_dim=interval_index_dim, hidden_dim=interval_hidden_dim)
        self._attention = None
        self._state = None
        self.model.register_forward_hook(self._capture_state_attention)

    def _capture_state_attention(self, module, inputs, output):
        self._state = output[0]
        self._attention = output[1]

    def forward(self, dcur, qtest=False, train=False, **kwargs):
        self._attention = None
        self._state = None
        result = super().forward(dcur, qtest=qtest, train=train, **kwargs)
        if not train:
            return result
        predictions, y2, y3, regularization = result
        if self.training and self.window_reg_weight > 0:
            valid = _causal_concept_mask(dcur)
            balance_loss = self.auxiliary(valid)
            regularization = regularization + self.window_reg_weight * balance_loss
        return (predictions, y2, y3, regularization)

class CL4KTIntervalTargets(IntervalTargetMixin, CL4KT):

    def __init__(self, num_skills, num_questions, seq_len, *args, window_reg_weight=0.01, interval_step=2, interval_max_seq_len=200, interval_index_dim=32, interval_hidden_dim=128, **kwargs):
        self.window_reg_weight = float(window_reg_weight)
        super().__init__(num_skills, num_questions, seq_len, **kwargs)
        self._init_interval_targets(d_model=int(self.hidden_size), interval_step=interval_step, max_seq_len=interval_max_seq_len, index_dim=interval_index_dim, hidden_dim=interval_hidden_dim)
        self._attention = None
        self._state = None
        self._interval_aux = None
        self.knoweldge_retriever[-1].register_forward_hook(self._capture_state_attention)

    def _capture_state_attention(self, module, inputs, output):
        self._state = output[0]
        self._attention = output[1]

    def forward(self, batch):
        self._attention = None
        self._state = None
        out_dict = super().forward(batch)
        self._interval_aux = next(self.interval_target_net.parameters()).new_zeros(())
        if self.training and self.window_reg_weight > 0:
            valid = batch['attention_mask'][2].bool()
            balance_loss = self.auxiliary(valid)
            self._interval_aux = self.window_reg_weight * balance_loss
        return out_dict

    def loss(self, feed_dict, out_dict):
        loss, n, correct = super().loss(feed_dict, out_dict)
        if self.training and self._interval_aux is not None:
            loss = loss + self._interval_aux
        return (loss, n, correct)

class AKTIntervalRescaled(ExplicitIntervalRescaleMixin, AKTIntervalTargets):

    def __init__(self, *args, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rescaling(rescale_eps, rescale_alpha, rescale_beta)
        self._rescale_dropout_layer = self.model.blocks_2[-1].masked_attn_head.dropout

    def forward(self, q_data, target, pid_data=None, qtest=False):
        self._attention, self._state = (None, None)
        AKT.forward(self, q_data, target, pid_data=pid_data, qtest=False)
        valid = q_data.ne(0)
        auxiliary = self._set_auxiliary(valid)
        scale = self._interval_scale(valid)
        self._attention, self._state = (None, None)
        if scale is None:
            result = AKT.forward(self, q_data, target, pid_data=pid_data, qtest=qtest)
        else:
            with self._rescaled_dropout(self._rescale_dropout_layer, scale):
                result = AKT.forward(self, q_data, target, pid_data=pid_data, qtest=qtest)
        if qtest:
            predictions, regularization, features = result
            return (predictions, regularization + auxiliary, features)
        predictions, regularization = result
        return (predictions, regularization + auxiliary)

class SAKTIntervalRescaled(ExplicitIntervalRescaleMixin, SAKTIntervalTargets):

    def __init__(self, *args, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rescaling(rescale_eps, rescale_alpha, rescale_beta)

    @contextmanager
    def _rescaled_sakt_attention(self, scale):
        attention_module = self.blocks[-1].attn
        original = attention_module.forward

        def forward(module, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
            if module.batch_first or key_padding_mask is not None or module.bias_k is not None or (module.bias_v is not None):
                raise RuntimeError('Unsupported SAKT MultiheadAttention configuration')
            length, batch, embed = query.shape
            source_length = key.shape[0]
            wq, wk, wv = module.in_proj_weight.chunk(3, dim=0)
            if module.in_proj_bias is None:
                bq, bk, bv = (None, None, None)
            else:
                bq, bk, bv = module.in_proj_bias.chunk(3)
            q = F.linear(query, wq, bq)
            k = F.linear(key, wk, bk)
            v = F.linear(value, wv, bv)
            heads = module.num_heads
            dim = embed // heads
            q = q.transpose(0, 1).reshape(batch, length, heads, dim).transpose(1, 2)
            k = k.transpose(0, 1).reshape(batch, source_length, heads, dim).transpose(1, 2)
            v = v.transpose(0, 1).reshape(batch, source_length, heads, dim).transpose(1, 2)
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
            if attn_mask is not None:
                shaped_mask = attn_mask.view(1, 1, length, source_length)
                if attn_mask.dtype == torch.bool:
                    logits = logits.masked_fill(shaped_mask, float('-inf'))
                else:
                    logits = logits + shaped_mask
            attention = torch.softmax(logits, dim=-1)
            attention = attention * scale.unsqueeze(1)
            denominator = attention.sum(dim=-1, keepdim=True)
            attention = torch.where(denominator.gt(self.rescale_eps), attention / denominator.clamp_min(self.rescale_eps), attention)
            self.last_rescaled_row_sum_error = attention.sum(dim=-1).sub(1.0).abs().masked_select(denominator.squeeze(-1).gt(self.rescale_eps)).max().detach() if denominator.gt(self.rescale_eps).any() else attention.new_zeros(())
            output = torch.matmul(F.dropout(attention, p=module.dropout, training=module.training), v)
            output = output.transpose(1, 2).reshape(batch, length, embed).transpose(0, 1)
            output = module.out_proj(output)
            if not need_weights:
                returned = None
            elif average_attn_weights:
                returned = attention.mean(dim=1)
            else:
                returned = attention
            return (output, returned)
        attention_module.forward = types.MethodType(forward, attention_module)
        try:
            yield
        finally:
            attention_module.forward = original

    def forward(self, q, r, qry, qtest=False):
        self._attention, self._state = (None, None)
        SAKT.forward(self, q, r, qry, qtest=False)
        valid = qry.ne(0)
        self.window_aux_loss = self._set_auxiliary(valid)
        scale = self._interval_scale(valid)
        self._attention, self._state = (None, None)
        if scale is None:
            return SAKT.forward(self, q, r, qry, qtest=qtest)
        with self._rescaled_sakt_attention(scale):
            return SAKT.forward(self, q, r, qry, qtest=qtest)

class FoLiBiIntervalRescaled(ExplicitIntervalRescaleMixin, FoLiBiIntervalTargets):

    def __init__(self, *args, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rescaling(rescale_eps, rescale_alpha, rescale_beta)
        self._rescale_dropout_layer = self.model.blocks_2[-1].masked_attn_head.dropout

    def forward(self, q_data, target, pid_data=None, qtest=False):
        self._attention, self._state = (None, None)
        folibiKT.forward(self, q_data, target, pid_data=pid_data, qtest=False)
        valid = q_data.ne(0)
        auxiliary = self._set_auxiliary(valid)
        scale = self._interval_scale(valid)
        self._attention, self._state = (None, None)
        if scale is None:
            result = folibiKT.forward(self, q_data, target, pid_data=pid_data, qtest=qtest)
        else:
            with self._rescaled_dropout(self._rescale_dropout_layer, scale):
                result = folibiKT.forward(self, q_data, target, pid_data=pid_data, qtest=qtest)
        if qtest:
            predictions, regularization, features = result
            return (predictions, regularization + auxiliary, features)
        predictions, regularization = result
        return (predictions, regularization + auxiliary)

class RobustKTIntervalRescaled(ExplicitIntervalRescaleMixin, RobustKTIntervalTargets):

    def __init__(self, *args, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rescaling(rescale_eps, rescale_alpha, rescale_beta)
        self._rescale_dropout_layer = self.model.blocks_2[-1].masked_attn_head.dropout

    def forward(self, q_data, target, pid_data=None, qtest=False):
        self._attention, self._state = (None, None)
        Robustkt.forward(self, q_data, target, pid_data=pid_data, qtest=False)
        valid = q_data.ne(0)
        auxiliary = self._set_auxiliary(valid)
        scale = self._interval_scale(valid)
        self._attention, self._state = (None, None)
        if scale is None:
            result = Robustkt.forward(self, q_data, target, pid_data=pid_data, qtest=qtest)
        else:
            with self._rescaled_dropout(self._rescale_dropout_layer, scale):
                result = Robustkt.forward(self, q_data, target, pid_data=pid_data, qtest=qtest)
        if qtest:
            predictions, regularization, features = result
            return (predictions, regularization + auxiliary, features)
        predictions, regularization = result
        return (predictions, regularization + auxiliary)

class SimpleKTIntervalRescaled(ExplicitIntervalRescaleMixin, SimpleKTIntervalTargets):

    def __init__(self, *args, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rescaling(rescale_eps, rescale_alpha, rescale_beta)
        self._rescale_dropout_layer = self.model.blocks_2[-1].masked_attn_head.dropout

    def forward(self, dcur, qtest=False, train=False):
        self._attention, self._state = (None, None)
        simpleKT.forward(self, dcur, qtest=qtest, train=train)
        valid = _causal_concept_mask(dcur)
        self.window_aux_loss = self._set_auxiliary(valid)
        scale = self._interval_scale(valid)
        self._attention, self._state = (None, None)
        if scale is None:
            return simpleKT.forward(self, dcur, qtest=qtest, train=train)
        with self._rescaled_dropout(self._rescale_dropout_layer, scale):
            return simpleKT.forward(self, dcur, qtest=qtest, train=train)

class SparseKTIntervalRescaled(ExplicitIntervalRescaleMixin, SparseKTIntervalTargets):

    def __init__(self, *args, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rescaling(rescale_eps, rescale_alpha, rescale_beta)
        self._rescale_dropout_layer = self.model.blocks_2[-1].masked_attn_head.dropout

    def forward(self, dcur, qtest=False, train=False, **kwargs):
        self._attention, self._state = (None, None)
        sparseKT.forward(self, dcur, qtest=qtest, train=train, **kwargs)
        valid = _causal_concept_mask(dcur)
        auxiliary = self._set_auxiliary(valid)
        scale = self._interval_scale(valid)
        self._attention, self._state = (None, None)
        if scale is None:
            result = sparseKT.forward(self, dcur, qtest=qtest, train=train, **kwargs)
        else:
            with self._rescaled_dropout(self._rescale_dropout_layer, scale):
                result = sparseKT.forward(self, dcur, qtest=qtest, train=train, **kwargs)
        if not train:
            return result
        predictions, y2, y3, regularization = result
        return (predictions, y2, y3, regularization + auxiliary)

class CL4KTIntervalRescaled(ExplicitIntervalRescaleMixin, CL4KTIntervalTargets):

    def __init__(self, *args, rescale_eps=1e-08, rescale_alpha=1.0, rescale_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rescaling(rescale_eps, rescale_alpha, rescale_beta)
        self._rescale_dropout_layer = self.knoweldge_retriever[-1].masked_attn_head.dropout

    def forward(self, batch):
        self._attention, self._state = (None, None)
        CL4KT.forward(self, batch)
        valid = batch['attention_mask'][2].bool()
        self._interval_aux = self._set_auxiliary(valid)
        scale = self._interval_scale(valid)
        self._attention, self._state = (None, None)
        if scale is None:
            return CL4KT.forward(self, batch)
        with self._rescaled_dropout(self._rescale_dropout_layer, scale):
            return CL4KT.forward(self, batch)
