from dataclasses import replace
import json

import pytest

from test_protocol_b_reference import api, reference_inputs, registered, inventory, reference_template


def test_registration_snapshot_must_match_file_content_and_hash(api, registered):
    table = registered["table"]
    value = json.loads(table.serialized)
    identifier = registered["context"].rows("val").iloc[0].image_id
    value["entries"][identifier]["query_canvas_to_bank"][0][2] += 5
    changed = replace(table, serialized=json.dumps(value, sort_keys=True))
    with pytest.raises(ValueError):
        api.ProtocolBReferenceDataset(registered["context"], changed, role="val", reference_enabled=False,
                                      augment={"enabled": False})
    changed_hash = replace(table, authority_sha256="0" * 64)
    with pytest.raises(ValueError):
        api.ProtocolBReferenceDataset(registered["context"], changed_hash, role="val", reference_enabled=False,
                                      augment={"enabled": False})
