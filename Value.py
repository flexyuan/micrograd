from math import exp


class Value:
    def __init__(self, data, _children=[], _op="") -> None:
        self.grad = 0
        self.data = data
        self._children = _children
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        children = [self, other]

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out = Value(self.data + other.data, children, "+")
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def _neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        children = [self]
        out = Value(self.data**other, children, "pow")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        children = [self, other]
        out = Value(self.data * other.data, children, "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def sigmoid(self):
        x = self.data
        t = (exp(2 * x) - 1) / (exp(2 * x) + 1)
        out = Value(t, [self], "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
