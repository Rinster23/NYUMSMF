import torch
import pdb

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

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    ## generate activation function and its derivative
    def gen_function(self, function):
        functions = {'identity': lambda x:x, 'relu':lambda x: torch.maximum(x, torch.zeros_like(x)),\
                    'sigmoid': lambda x: 1/(1+torch.exp(-x))}
        d_functions = {'identity': lambda x:torch.ones_like(x), 'relu':lambda x:x.apply_(lambda y:0 if y<=0 else 1),\
                    'sigmoid': lambda x: 1/(1+torch.exp(-x)) * (1-1/(1+torch.exp(-x)))}

        if not(self.f_function in functions and self.g_function in functions):
            print('There is something in this functions: {} and {}'.format(self.f_function, self.g_function))
            exit(1)
        else:
            return (functions[function], d_functions[function])

    ## generate derivatives
    def gen_derivative(self, function, x):
        d = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
        derivatives = self.gen_function(function)[1](x)
        for i in range(derivatives.shape[0]):
            for j in range(derivatives.shape[1]):
                d[i,j,j] = derivatives[i,j]
        return d
            

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']
        self.cache['x'] = x
        self.cache['s1'] = torch.mm(W1, x.t()).t() + b1.unsqueeze(0)
        self.cache['a1'] = self.gen_function(self.f_function)[0](self.cache['s1'])
        self.cache['s2'] = torch.mm(W2, self.cache['a1'].t()).t() + b2.unsqueeze(0)
        self.cache['a2'] = self.gen_function(self.g_function)[0](self.cache['s2'])
        return self.cache['a2']


    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        x, s1, a1, s2, a2 = self.cache['x'], self.cache['s1'], self.cache['a1'], self.cache['s2'], self.cache['a2']
        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']
        dy_hatds2 = self.gen_derivative(self.g_function, s2)
        dJdW2 = torch.bmm(torch.bmm(a1.unsqueeze(2), dJdy_hat.unsqueeze(1)), dy_hatds2)
        dJdb2 = torch.bmm(dJdy_hat.unsqueeze(1), dy_hatds2).squeeze(1)
        W2_batch = W2.unsqueeze(0).repeat(x.shape[0], 1, 1)
        dJdW1 = torch.bmm(torch.bmm(torch.bmm(torch.bmm(x.unsqueeze(2), dJdy_hat.unsqueeze(1)), dy_hatds2), W2_batch),\
                self.gen_derivative(self.f_function, s1))
        dJdb1 = torch.bmm(torch.bmm(torch.bmm(dJdy_hat.unsqueeze(1), dy_hatds2), W2_batch),\
                self.gen_derivative(self.f_function, s1)).squeeze(1)
        self.grads['dJdW2'] = torch.sum(dJdW2, axis=0).t()
        self.grads['dJdb2'] = torch.sum(dJdb2, axis=0)
        self.grads['dJdW1'] = torch.sum(dJdW1, axis=0).t()
        self.grads['dJdb1'] = torch.sum(dJdb1, axis=0)

    
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
    loss = torch.mean((y - y_hat)*(y - y_hat))
    dJdy_hat = (y_hat - y)*2/y_hat.numel()
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
    loss = torch.mean(-y*torch.log(y_hat) - (1-y)*torch.log(1-y_hat))
    dJdy_hat = ((1-y)/(1-y_hat) - y/y_hat)/y_hat.numel()
    return loss, dJdy_hat











