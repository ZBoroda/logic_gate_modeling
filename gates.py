#! /usr/bin/python3
import enum


class LogicGateType(enum.Enum):
    INPUT = 1
    NAND = 2


class logic_gate:
    def __init__(self, type, inputs):
        self.type = type
        self.inputs = inputs
        self.result = False

    def process(self, in_vals):
        if self.type == LogicGateType.INPUT:
            self.result = in_vals[0]
        elif self.type == LogicGateType.NAND:
            self.result = not (in_vals[0] and in_vals[1])
        return self.result


def evaluate(gates, sources):
    for gate in gates:
        # print (gate.inputs)
        if gate.type == LogicGateType.INPUT:
            in_vals = (sources[gate.inputs[0]],)
        elif gate.type == LogicGateType.NAND:
            in_vals = (gates[gate.inputs[0]].result, gates[gate.inputs[1]].result)

        result = gate.process(in_vals)

    return result


gatesList = [
    logic_gate(LogicGateType.INPUT, (0,)),
    logic_gate(LogicGateType.INPUT, (1,)),
    logic_gate(LogicGateType.INPUT, (2,)),
    logic_gate(LogicGateType.INPUT, (3,)),
    logic_gate(LogicGateType.NAND, (0, 1)),  # gate5
    logic_gate(LogicGateType.NAND, (1, 2)),  # gate6
    logic_gate(LogicGateType.NAND, (4, 5)),  # gate7
    logic_gate(LogicGateType.NAND, (5, 3)),  # gate8
    logic_gate(LogicGateType.NAND, (6, 7)),  # gate9
]

inputs = [True, True, False, True]

print("The result is " + str(evaluate(gatesList, inputs)))

# for gate in gatesList:
# 	print(gate)
