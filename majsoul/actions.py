from enum import StrEnum


class AgentActionType(StrEnum):
    """Actions the agent may deliberately choose at a decision window.

    These are not observations. In particular, PASS is an agent response and
    must never be reconstructed as a table-state event without interaction
    evidence.
    """

    DISCARD = "DISCARD"
    RIICHI = "RIICHI"
    CHI = "CHI"
    PON = "PON"
    KAN = "KAN"
    KITA = "KITA"
    RON = "RON"
    TSUMO = "TSUMO"
    PASS = "PASS"
