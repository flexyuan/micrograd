from Value import Value
import random


class Neuron:

    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x) -> None:
        z = sum((x1 * w1 for x1, w1 in zip(x, self.w)), self.b)
        return z.sigmoid()
    
    def parameters(self):
        r = [i for i in self.w]
        r.append(self.b)
        return r


class Layer:
    def __init__(self, nin, nout) -> None:
        self.nin = nin
        self.nout = nout
        self.neurons: list[Neuron] = [Neuron() for i in range(nout)]

    def __call__(self, x):
        return [l(x) for l in self.neurons]


class MLP:
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        result = []
        for i in self.neurons:
            result.extend(i.parameters())
