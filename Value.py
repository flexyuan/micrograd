from math import exp


class Value:
    def __init__(self, data, _children=[], _op="") -> None:
        self.grad = 0
        self.data = data
        self._children = _children
        self._op = _op
        self._backprop = None
        self._backward = None

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
        return Value(
            -self.data,
        )

    def __mult__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        children = [self, other]
        out = Value(self.data * other.data, children, "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmult__(self, other):
        return self * other

    def backprop(self):
        for c in self._children:
            c.backprop()

    def sigmoid(self):
        d = 1 / (1 + exp(self.data))
        v = Value(d, [self], "sigmoid")

        def backward():
            self.grad += v.grad * (1 - d) * d

        v._backward = backward
        return

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
