import torch
import torch.nn as nn
import numpy as np


class MLP:
    def __init__(
            self,
            linear_1_in_features,
            linear_1_out_features,
            f_function,
            linear_2_in_features,
            linear_2_out_features,
            g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function
        self.ly_0 = linear_1_in_features
        self.ly_1 = linear_1_out_features
        self.ly_2 = linear_2_out_features
        self.batch_size = None

        self.parameters = dict(
            W1=torch.randn(linear_1_out_features, linear_1_in_features),
            b1=torch.randn(linear_1_out_features),
            W2=torch.randn(linear_2_out_features, linear_2_in_features),
            b2=torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1=torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1=torch.zeros(linear_1_out_features),
            dJdW2=torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2=torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.batch_size = x.size(0)
        self.cache['x'] = x
        self.cache['W2'] = self.parameters['W2']
        activation_fun = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'identity': nn.Identity()}

        y = self.parameters['W1'] @ x.t() + self.parameters['b1'].view(-1, 1)
        self.cache['S1'] = y.t()  # (batch_size, linear_2_out_features)
        y = activation_fun[self.f_function](y)
        self.cache['alpha_1'] = y.t()  # (batch_size, linear_1_out_features)
        y = self.parameters['W2'] @ y + self.parameters['b2'].view(-1, 1)
        self.cache['S2'] = y.t()  # (batch_size, linear_2_out_features)
        y = activation_fun[self.g_function](y)
        return y.t()  # (batch_size, linear_2_out_features)

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        x = self.cache['x']
        S1 = self.cache['S1']
        alpha_1 = self.cache['alpha_1']  # (batch_size,linear_1_out_features)
        W2 = self.cache['W2']
        S2 = self.cache['S2']  # (batch_size, linear_2_out_features)

        y_hat_S2 = {
            'relu': [torch.diag(torch.where(S2[i] > 0, 1.0, 0.0)) for i in range(self.batch_size)],
            'sigmoid': [torch.diag(nn.Sigmoid()(S2[i]) * (1 - nn.Sigmoid()(S2[i]))) for i in range(self.batch_size)]
        }

        alpha_1_S1 = {
            'relu': [torch.diag(torch.where(S1[i] > 0, 1.0, 0.0)) for i in range(self.batch_size)],
            'sigmoid': [torch.diag(nn.Sigmoid()(S1[i]) * (1 - nn.Sigmoid()(S1[i]))) for i in range(self.batch_size)]
        }
        if self.g_function == 'identity':
            self.grads['dJdW2'] = (alpha_1.t() @ dJdy_hat / self.batch_size).t()
            self.grads['dJdb2'] = dJdy_hat.mean(axis=0)
        else:
            temp1 = torch.zeros(self.ly_1, self.ly_2)
            temp2 = torch.zeros(1, self.ly_2)
            for i in range(self.batch_size):
                dy_hatdS2 = y_hat_S2[self.g_function][i]
                temp1 = temp1 + alpha_1[i].view(-1, 1) @ dJdy_hat[i].view(1, -1) @ dy_hatdS2
                temp2 = temp2 + dJdy_hat[i].view(1, -1) @ dy_hatdS2
            self.grads['dJdW2'] = temp1.t() / self.batch_size
            self.grads['dJdb2'] = temp2.squeeze() / self.batch_size

        if self.g_function == 'identity':
            if self.f_function == 'identity':
                self.grads['dJdW1'] = (x.t() @ dJdy_hat @ W2).t() / self.batch_size
                temp = dJdy_hat @ W2
                self.grads['dJdb1'] = temp.mean(axis=0)
            else:
                temp1 = torch.zeros(self.ly_0, self.ly_1)
                temp2 = torch.zeros(1, self.ly_1)
                d = alpha_1_S1[self.f_function]
                for i in range(self.batch_size):
                    dalpha_1dS1 = d[i]
                    temp1 = temp1 + x[i].view(-1, 1) @ dJdy_hat[i].view(1, -1) @ W2 @ dalpha_1dS1
                    temp2 = temp2 + dJdy_hat[i].view(1, -1) @ W2 @ dalpha_1dS1
                self.grads['dJdW1'] = temp1.t() / self.batch_size
                self.grads['dJdb1'] = temp2.squeeze() / self.batch_size
        else:
            if self.f_function == 'identity':
                temp1 = torch.zeros(self.ly_0, self.ly_1)
                temp2 = torch.zeros(1, self.ly_1)
                for i in range(self.batch_size):
                    dy_hatdS2 = y_hat_S2[self.g_function][i]
                    temp1 = temp1 + x[i].view(-1, 1) @ dJdy_hat[i].view(1, -1) @ dy_hatdS2 @ W2
                    temp2 =    temp2 + dJdy_hat[i].view(1, -1) @ dy_hatdS2 @ W2
                self.grads['dJdW1'] = temp1.t() / self.batch_size
                self.grads['dJdb1'] = temp2.squeeze() / self.batch_size
            else:
                temp1 = torch.zeros(self.ly_0, self.ly_1)
                temp2 = torch.zeros(1, self.ly_1)
                for i in range(self.batch_size):
                    dy_hatdS2 = y_hat_S2[self.g_function][i]
                    dalpha_1dS1 = alpha_1_S1[self.f_function][i]
                    temp1 = temp1 + x[i].view(-1, 1) @ dJdy_hat[i].view(1, -1) @ dy_hatdS2 @ W2 @ dalpha_1dS1
                    temp2 = temp2 + dJdy_hat[i].view(1, -1) @ dy_hatdS2 @ W2 @ dalpha_1dS1
                self.grads['dJdW1'] = temp1.t() / self.batch_size
                self.grads['dJdb1'] = temp2.squeeze() / self.batch_size

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    batch_size = y.size(0)
    M = y.size(1)
    loss = (y - y_hat) * (y - y_hat)
    loss = loss.sum() / batch_size / M
    dJdy_hat = 2 * (y_hat - y) / M
    return loss, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    batch_size = y.size(0)
    M = y.size(1)
    loss = (-y * torch.where(torch.log(y_hat) < -100, -100, torch.log(y_hat))
            - (1 - y) * torch.where(torch.log(1 - y_hat) < -100, -100, torch.log(1 - y_hat)))
    loss = loss.sum() / batch_size / M
    dJdy_hat = (-y / torch.where(y_hat < np.exp(-100), np.exp(-10), y_hat) +
                (1 - y) / torch.where(1 - y_hat < np.exp(-100), np.exp(-10), 1 - y_hat)) / M
    return loss, dJdy_hat

    # return loss, dJdy_hat
