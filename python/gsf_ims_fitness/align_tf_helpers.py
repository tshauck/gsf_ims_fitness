"""
Helper functions for aligning transcription factors and ligands.
"""


def align_tf_from_ligand(lig):
    # TODO: edit this to use sample_plate_map
    if lig == "IPTG":
        return "LacI"
    if lig == "1S-TIQ":
        return "RamR"
    if lig == "Van":
        return "VanR"

    raise ValueError(f"Unexpected ligand: {lig}")


def align_ligand_from_tf(tf):
    # TODO: edit this to use sample_plate_map
    if tf == "LacI":
        return "IPTG"
    if tf == "RamR":
        return "1S-TIQ"
    if tf == "VanR":
        return "Van"

    raise ValueError(f"Unexpected transcription factor: {tf}")


def align_tf_from_RS_name(rs):
    if "LacI" in rs:
        return "LacI"
    if "RamR" in rs:
        return "RamR"
    if "VanR" in rs:
        return "VanR"

    raise ValueError(f"Unexpected repressor name: {rs}")
