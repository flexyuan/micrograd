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
        d = 1 / (1 + exp(self.data))
        v = Value(d, [self], "sigmoid")

        def backward():
            self.grad += v.grad * (1 - d) * d

        v._backward = backward
        return v

    def backward(self):
        topo = []
        q: list[Value] = [self]
        while len(q) != 0:
            v = q.pop(0)
            topo.append(v)
            q.extend(v._children)
        self.grad = 1.0
        for p in topo:
            p._backward()


    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
