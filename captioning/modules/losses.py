import torch
import torch.nn as nn
from ..utils.rewards import get_scores, get_self_cider_scores


class RewardCriterion(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018).
    """
    def __init__(self, opt):
        super(RewardCriterion, self).__init__()
        self.opt = opt
    def forward(self, input, seq, data_gts):
        """
        Input is either logits or log softmax
        """
        out = {}
        batch_size = input.size(0)# batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)
        assert seq_per_img == self.opt.train_sample_n, seq_per_img
        mask = (seq>0).to(input)
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)
        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores #.mean()
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
        scores = scores - baseline
        output = - input * mask * scores.view(-1, 1)
        output = torch.sum(output) / torch.sum(mask)

        out['loss'] = output
        return out

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        # Average over each token
        output = torch.sum(output) / torch.sum(mask)

        return output

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None
        
    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()