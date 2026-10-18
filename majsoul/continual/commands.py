import re


_LABEL_COMMAND = re.compile(
    r"^/?label\s+(?P<case>(?:case_)?\d{1,6})\s+(?P<label>[A-Za-z0-9_.:-]+)$",
    re.IGNORECASE,
)
_REJECT_COMMAND = re.compile(
    r"^/?reject\s+(?P<case>(?:case_)?\d{1,6})\s+(?P<reason>\S(?:.*\S)?)$",
    re.IGNORECASE,
)


def _case_id(raw_case: str) -> str:
    number = int(raw_case.lower().removeprefix("case_"))
    if number <= 0:
        raise ValueError("Case id must be positive")
    return f"case_{number:06d}"


def parse_label_command(text: str) -> tuple[str, str]:
    """Parse one explicit human label command; reject surrounding chat text."""
    match = _LABEL_COMMAND.fullmatch(text.strip())
    if match is None:
        raise ValueError("Expected: /label <case_id> <label>")
    return _case_id(match.group("case")), match.group("label")


def parse_review_command(text: str) -> tuple[str, str, str]:
    """Parse exactly one explicit label or rejection command."""
    stripped = text.strip()
    label = _LABEL_COMMAND.fullmatch(stripped)
    if label is not None:
        return "label", _case_id(label.group("case")), label.group("label")
    reject = _REJECT_COMMAND.fullmatch(stripped)
    if reject is not None:
        reason = reject.group("reason").strip()
        if len(reason) > 500:
            raise ValueError("Rejection reason is too long")
        return "reject", _case_id(reject.group("case")), reason
    raise ValueError("Expected: /label <case_id> <label> or /reject <case_id> <reason>")
