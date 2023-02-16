# Taken from https://github.com/facebookresearch/ParlAI/blob/main/projects/cringe/cringe_loss.py
"""
Transformer Agent with a contrastive loss.
"""

from typing import Optional, Dict, Union

import torch

from torch.nn import CrossEntropyLoss
from torch.distributions.categorical import Categorical


class ContrastiveCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        ct_loss_weight=1.0,
        num_pos_predictions=1,
        detach_positives_during_ct=False,
        train_ct_on_positive_examples=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ct_loss_weight = ct_loss_weight
        self.num_pos_predictions = num_pos_predictions
        self.detach_positives_during_ct = detach_positives_during_ct
        self.train_ct_on_positive_examples = train_ct_on_positive_examples

    def __call__(self, x, y, classifier_labels=None, **kwargs):
        if classifier_labels is None:
            classifier_labels = -torch.ones_like(y).to(y.device)

        # turn no-class provided label (-1) into positive label (1)
        classifier_labels_ce = torch.abs(classifier_labels)

        if self.train_ct_on_positive_examples:
            # no-class (-1 to 0), positive (1 to 1), negative (0 to 1)
            classifier_labels_ct = torch.clamp(classifier_labels + 1, max=1)
        else:
            # no-class (-1 to 0), positive (1 to 0), negative (0 to 1)
            classifier_labels_ct = torch.abs(torch.abs(classifier_labels) - 1)

        ce_loss = super().__call__(x, y, **kwargs)
        # multiply with classifier labels to not train with negative feedback (0)
        ce_loss *= classifier_labels_ce

        # compute the contrastive loss part for the negative labels
        # first, get the positives as the top predictions != target
        preds = torch.topk(x, k=self.num_pos_predictions + 1, axis=-1)
        y_rep = y.unsqueeze(1).repeat(1, self.num_pos_predictions + 1)
        logits = preds.values - (preds.indices == y_rep) * 1e10

        # if the positive is not in the first k predictions, mask out
        # the final (k+1)'s logit
        prediction_mask = torch.cat(
            (
                torch.zeros_like(logits)[:, :-1],
                torch.abs((preds.indices == y_rep).sum(-1).unsqueeze(1) - 1),
            ),
            1,
        )
        logits -= prediction_mask * 1e10

        # Sample from the categorical distribution of the top-k predictions
        # (with the label masked out).
        preds_dist = Categorical(logits=logits)
        idx_sample = preds_dist.sample()
        sample_preds_values = preds.values[torch.arange(x.shape[0]), idx_sample]

        if self.detach_positives_during_ct:
            sample_preds_values = sample_preds_values.detach()

        # concatenate the logits of the preds with the actual label's logits
        x_target = x[torch.arange(x.shape[0]), y]
        x_ct = torch.cat([x_target.unsqueeze(1), sample_preds_values.unsqueeze(1)], -1)
        # get the y's for the x_ct (the correct label is index 0 if
        # the target is positive and index 1 if the target is negative)
        y_ct = torch.abs(torch.abs(classifier_labels) - 1).type(y.dtype).to(x_ct.device)
        # y_ct = (torch.ones(y.shape) * ).type(y.dtype).to(x_ct.device)
        # compute the contrastive loss as cross entropy loss between x_ct, y_ct
        ct_loss = super().__call__(x_ct, y_ct, **kwargs)
        ct_loss *= classifier_labels_ct

        # remove loss from ignore index
        notnull = y.ne(self.ignore_index)
        ce_loss *= notnull
        ct_loss *= notnull

        loss = ce_loss + self.ct_loss_weight * ct_loss

        return loss, ce_loss, ct_loss
