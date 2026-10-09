"""Loss = BCE(score) + 3 * (BCE + Dice)(mask)."""


def referral_loss(score_logit, mask_logit, label, mask, mask_weight=3.0):
    raise NotImplementedError
