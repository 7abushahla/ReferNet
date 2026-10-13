"""Diagnostic-only head dropout, reconstructed explicitly before state loading.

Parameter keys and initialization tensors remain unchanged. Regularization
metadata belongs in the diagnostic receipt; legacy checkpoint validation is
not an authorization path for this architecture.
"""

from copy import deepcopy

from torch import nn

from .direct_reference import DirectReferenceClassifier


ARCHITECTURE_ID = "anchor_direct_reference_ia18_head_dropout_v1"


class _DropoutInput:
    def __init__(self, dropout):
        self.dropout = dropout

    def __call__(self, module, arguments):
        return (self.dropout(arguments[0]), *arguments[1:])


def regularize_model(model: DirectReferenceClassifier, p: float) -> DirectReferenceClassifier:
    """Install equal OFF/ON readout dropout once, without replacing parameters.

Reconstruction requires the same fresh model and rate, followed by strict
state_dict loading. Install on each reconstructed model, not on a legacy
checkpoint interpreted as this architecture.
"""
    if not isinstance(model, DirectReferenceClassifier):
        raise ValueError("regularization requires a DirectReferenceClassifier")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or p not in (0.0, 0.1, 0.3):
        raise ValueError("dropout p must be 0.0, 0.1, or 0.3")
    if hasattr(model, "global_dropout") or hasattr(model, "local_dropout") or "regularization" in model.initialization_provenance:
        raise ValueError("head dropout is already installed")
    if model.architecture_id != DirectReferenceClassifier.architecture_id:
        raise ValueError("original direct-reference architecture required")
    classifier = model.base.classifier
    if not isinstance(classifier, nn.Sequential) or len(classifier) != 3 or not isinstance(classifier[1], nn.ReLU):
        raise ValueError("original global Linear/ReLU/Linear head required")
    global_linear = classifier[2]
    local_linear = model.adapter.correction_head
    if not isinstance(global_linear, nn.Linear) or (global_linear.in_features, global_linear.out_features) != (128, 1):
        raise ValueError("global final linear must be 128 to 1")
    if not isinstance(local_linear, nn.Linear) or (local_linear.in_features, local_linear.out_features) != (64, 1):
        raise ValueError("local final linear must be 64 to 1")
    provenance = deepcopy(model.initialization_provenance)
    provenance.update(architecture_id=ARCHITECTURE_ID, regularization={
        "policy": "head_readout_dropout_v1", "p": float(p),
        "global_placement": "after_relu_before_final_linear",
        "local_placement": "after_masked_pool_before_correction_linear",
    })
    model.global_dropout = nn.Dropout(float(p)).train(model.training)
    model.local_dropout = nn.Dropout(float(p)).train(model.training)
    global_linear.register_forward_pre_hook(_DropoutInput(model.global_dropout))
    local_linear.register_forward_pre_hook(_DropoutInput(model.local_dropout))
    model.architecture_id = ARCHITECTURE_ID
    model.initialization_provenance = provenance
    return model
