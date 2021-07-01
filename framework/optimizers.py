#/******************************************
#*MIT License
#*
# *Copyright (c) [2021] [Luigi Fusco, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# */
import numpy as np

class GradientDescentOptimizer():
    def __init__(self, transform, loss, grad, learning_rate=0.01, alpha=1):
        self.learning_rate = learning_rate
        self.transform = transform
        self.loss = loss
        self.grad = grad
        self.alpha = alpha
        self.last_loss = None

    def _substep(self, fixed, moving):
        moved, image_transform_gradient = self.transform(moving, self.grad)
        self.last_loss, loss_gradient = self.loss(fixed, moved)

        return loss_gradient @ image_transform_gradient

    def step(self, fixed, moving):
        gradients = self._substep(fixed, moving).flatten()
        self.transform.parameters = (
            self.transform.parameters - self.learning_rate * gradients
        )
        self.learning_rate *= self.alpha


class OnePlusOneOptimizer():
    def __init__(self, transform, loss, rate):
        self.transform = transform
        self.loss = loss
        self.rate = rate

    def step(self, fixed, moving):
        parent_parameters = self.transform.parameters.copy()
        parent_score, _ = self.loss(fixed, self.transform(moving))
        self.transform.parameters += np.random.normal(0, self.rate)
        child_score, _ = self.loss(fixed, self.transform(moving))

        if parent_score < child_score:
            self.transform.parameters = parent_parameters
