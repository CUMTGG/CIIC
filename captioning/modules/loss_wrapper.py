import torch
from . import losses
from ..utils.rewards import init_scorer


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion(opt)


    def forward(self, fc_feats, att_feats, iod_feats, labels, masks, att_masks, iod_masks, gts, gt_indices,
                sc_flag):
        opt = self.opt

        out = {}
        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, iod_feats, labels[..., :-1], att_masks, iod_masks), labels[..., 1:],
                             masks[..., 1:])
        else:
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, iod_feats, att_masks, iod_masks,
                                                     opt={'sample_method': opt.train_sample_method,
                                                          'beam_size': opt.train_beam_size,
                                                          'sample_n': opt.train_sample_n},
                                                     mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            output = self.rl_crit(sample_logprobs, gen_result, gts)
            loss = output['loss']
            out['reward'] = output['reward']
        out['loss'] = loss
        return out
