"""Data quality checks at each layer transition."""


def validate_bronze() -> bool:
    # TODO: timestamp continuity, null rate checks
    return True


def validate_silver() -> bool:
    # TODO: range checks, type enforcement
    return True
