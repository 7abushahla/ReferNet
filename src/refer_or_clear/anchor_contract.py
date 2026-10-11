from dataclasses import dataclass


SCHEMA_VERSION = "anchor_pilot_v1"


@dataclass(frozen=True)
class PilotContract:
    track: str
    architecture_id: str
    data_contract: str
    channels: int
    height: int
    width: int
    batch_size: int
    eval_batch_size: int
    mask_weight: float
    bn_policy: str
    diagnostic_only: bool = False

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.channels, self.height, self.width


_CONTRACTS = {
    "R": PilotContract("R", "anchor_rgb_r50_v1", "raw_rgb_299_v1", 3, 299, 299, 32, 32, 0.0, "running"),
    "D": PilotContract("D", "anchor_direct_ia18_v1", "legacy_geometry_diagnostic", 4, 512, 800, 8, 1, 0.0, "batch_stats"),
    "D_running": PilotContract("D_running", "anchor_direct_running_ia18_v1", "legacy_geometry_diagnostic", 4, 512, 800, 8, 1, 0.0, "running", True),
    "D_running_pilot": PilotContract("D_running_pilot", "anchor_direct_running_ia18_pilot_v1", "legacy_geometry_diagnostic", 4, 512, 800, 8, 1, 0.0, "running"),
    "N": PilotContract("N", "anchor_absent_ia18_v1", "legacy_geometry_diagnostic", 4, 512, 800, 8, 8, 3.0, "running"),
}


def contract_for(track: str) -> PilotContract:
    try:
        return _CONTRACTS[track]
    except KeyError as error:
        raise ValueError(f"Unknown pilot track {track!r}; expected R, D, D_running, D_running_pilot or N") from error
