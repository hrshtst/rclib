from __future__ import annotations


class Ridge:
    def __init__(self, alpha, include_bias):
        self.alpha = alpha
        self.include_bias = include_bias


class Rls:
    def __init__(self, lambda_, delta, include_bias):
        self.lambda_ = lambda_
        self.delta = delta
        self.include_bias = include_bias


class Lms:
    def __init__(self, learning_rate, include_bias):
        self.learning_rate = learning_rate
        self.include_bias = include_bias
