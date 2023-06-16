def or_funct(x: [bool]) -> bool:
    return x[0] or x[1]


def and_funct(x: [bool]) -> bool:
    return x[0] and x[1]


def nor_funct(x: [bool]) -> bool:
    return not (x[0] or x[1])


def xor_funct(x: [bool]) -> bool:
    return x[0] != x[1]


def xnor_funct(x: [bool]) -> bool:
    return x[0] == x[1]