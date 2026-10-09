"""Extra training forgeries with the released SIDTD generator (crop-and-replace, inpaint-and-rewrite).

Generator: github.com/Oriolrt/SIDTD_Dataset, SIDTD/data/explore/fake_id_ex.py. Masks are exact because
the generator records the altered rectangle. Target about 5,000 forgeries over the MIDV-2020 templates.
"""


def generate(templates, annotations, n_per_identity):
    raise NotImplementedError
