"""Build the detector on ChangerEx from Open-CD (Apache-2.0), backbone in {resnet18, resnet50,
resnest50, mit_b1}, initialised from LEVIR-CD or ImageNet weights."""


def build_model(backbone, fusion, init="levir"):
    raise NotImplementedError
