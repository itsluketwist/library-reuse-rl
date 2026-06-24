"""A bitmask enum base class whose members carry their own float weight."""

import enum


class WeightedIntFlag(enum.IntFlag):
    """
    An IntFlag whose members carry a float `weight` alongside their bit value.

    Subclass this and declare members as plain weight literals (e.g. `FOO = 0.1`)
    instead of `enum.auto()` - the bit value (1, 2, 4, ...) is still auto-assigned
    in declaration order, so existing bits keep their values when a new member is
    added. This class itself defines no members, only the `__new__` override, since
    Python only allows subclassing an Enum that has none.
    """

    weight: float

    def __new__(cls, weight: float) -> "WeightedIntFlag":
        bit_value = 2 ** len(cls.__members__)
        member = int.__new__(cls, bit_value)
        member._value_ = bit_value
        member.weight = weight
        return member
