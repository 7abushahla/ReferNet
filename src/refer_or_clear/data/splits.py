"""Identity-grouped splits (DESIGN.md 3.5).

Protocol A: SIDTD's own split_shot file, 6 nationalities train, 4 test.
Protocol B: train on all 10 MIDV designs, test on FantasyID's official test split (300 genuine,
1,085 altered); the Arabic design is also reported alone. FantasyID's three captures of a card stay together.
"""


def protocol_a(records, split_shot_csv):
    raise NotImplementedError


def protocol_b(records, fantasyid_test_list):
    raise NotImplementedError
