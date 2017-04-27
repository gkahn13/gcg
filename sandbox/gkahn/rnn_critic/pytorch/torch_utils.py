import torch
import numpy as np

from sandbox.gkahn.rnn_critic.utils.utils import timeit

#################
### Functions ###
#################

def one_hot(size, index):
    """ https://github.com/mrdrozdov/pytorch-extras/blob/master/torch_extras/extras.py
        Creates a matrix of one hot vectors.
        ```
        import torch
        import torch_extras
        setattr(torch, 'one_hot', torch_extras.one_hot)
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ```
    """
    mask = torch.cuda.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, torch.autograd.Variable):
        ones = torch.autograd.Variable(torch.cuda.LongTensor(index.size()).fill_(1))
        mask = torch.autograd.Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret

def expand_dims(var, dim=0):
    """ Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).
        import torch
        import torch_extras
        setattr(torch, 'expand_dims', torch_extras.expand_dims)
        var = torch.range(0, 9).view(-1, 2)
        torch.expand_dims(var, 0).size()
        # (1, 5, 2)
    """
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)

##############
### Layers ###
##############

class FullyConnected(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, activation, output_activation=None):
        super(FullyConnected, self).__init__()

        ### create layers
        self._layers = []
        for i, output_dim in enumerate(layer_dims):
            layer = torch.nn.Linear(input_dim, output_dim).cuda()
            setattr(self, 'layer{0}'.format(i), layer) # so it is registered as a param
            self._layers.append(layer)
            input_dim = output_dim

        ### activations
        self._activations = [activation] * (len(layer_dims) - 1) + [output_activation]

        ### initialize weights
        for layer in self._layers:
            torch.nn.init.xavier_uniform(layer.weight, gain=np.sqrt(2.))
            torch.nn.init.normal(layer.bias, std=0.1)

    def forward(self, x):
        for i, (layer, activation) in enumerate(zip(self._layers, self._activations)):
            x = layer(x)
            if activation is not None:
                x = activation(x)

        return x

class CNN(torch.nn.Module):
    def __init__(self, input_shape, channel_dims, kernels, strides, activation):
        """
        :param input_shape: [channels, H, W]
        """
        super(CNN, self).__init__()
        self._input_shape = input_shape

        ### create layers
        self._layers = []
        input_channel = input_shape[0]
        for i, (output_channel, kernel, stride) in enumerate(zip(channel_dims, kernels, strides)):
            layer = torch.nn.Conv2d(input_channel, output_channel, kernel, stride).cuda()
            setattr(self, 'layer{0}'.format(i), layer) # to register as a param
            self._layers.append(layer)
            input_channel = output_channel

        ### activations
        self._activation = activation

        ### initialize weights
        for layer in self._layers:
            torch.nn.init.xavier_uniform(layer.weight, gain=np.sqrt(2.))
            torch.nn.init.constant(layer.bias, 0.1)

        ### calculate output dim
        tmp_forward = torch.nn.Sequential(*self._layers)(torch.autograd.Variable(torch.ones(1, *input_shape).cuda()))
        self._output_dim = int(np.prod(tmp_forward.size()[1:]))

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        """
        :param x: [H, W, channels]
        """
        x = x.resize(x.size(0), *self._input_shape)

        ### conv
        for layer in self._layers:
            x = self._activation(layer(x))

        ### flatten
        x = x.view(-1, self.output_dim)

        return x

###########
### RNN ###
###########

class MuxRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MuxRNNCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._rnn_cell = torch.nn.RNNCell(input_size * hidden_size, input_size * hidden_size)

    def forward(self, input_t, h_t):

        # import IPython; IPython.embed()

        # h_t_big = h_t.repeat(1, self._input_size)
        # h_tp1 = self._rnn_cell(h_t_big, h_t_big)[:, :self._hidden_size]

        batch_size = input_t.size(0)
        h_t_big = h_t.repeat(1, self._input_size)
        h_tp1_big = self._rnn_cell(h_t_big, h_t_big)
        h_tp1_big = h_tp1_big.resize(batch_size, self._input_size, self._hidden_size) # TODO order might be switched
        input_t_big = input_t.unsqueeze(2).repeat(1, 1, self._hidden_size)
        h_tp1 = (input_t_big * h_tp1_big).sum(1).squeeze(1)

        return h_tp1

class MuxLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MuxLSTMCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._rnn_cell = torch.nn.LSTMCell(input_size * hidden_size, input_size * hidden_size)

    def forward(self, input_t, hc_t):
        h_t, c_t = hc_t
        batch_size = input_t.size(0)

        h_t_big = h_t.repeat(1, self._input_size)
        c_t_big = c_t.repeat(1, self._input_size)
        h_tp1_big, c_tp1_big = self._rnn_cell(h_t_big, (h_t_big, c_t_big))
        h_tp1_big = h_tp1_big.resize(batch_size, self._input_size, self._hidden_size)  # TODO order might be switched
        c_tp1_big = c_tp1_big.resize(batch_size, self._input_size, self._hidden_size)  # TODO order might be switched

        input_t_big = input_t.unsqueeze(2).repeat(1, 1, self._hidden_size)
        h_tp1 = (input_t_big * h_tp1_big).sum(1).squeeze(1)
        c_tp1 = (input_t_big * c_tp1_big).sum(1).squeeze(1)

        return h_tp1, c_tp1


class MuxLSTMCellOLD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MuxLSTMCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._rnn_cells = []
        for i in range(input_size):
            rnn_cell = torch.nn.LSTMCell(hidden_size, hidden_size, bias=bias)
            setattr(self, 'rnn_cell{0}'.format(i), rnn_cell)
            self._rnn_cells.append(rnn_cell)

        import IPython; IPython.embed()

    def forward(self, input_t, hc_t):
        # timeit.start('total')
        # zero_input = torch.autograd.Variable(torch.zeros(input_t.size(0), self._hidden_size).cuda())
        # outputs = [cell(hc_t[1], hc_t) for cell in self._rnn_cells]
        outputs = [self._rnn_cells[0](hc_t[1], hc_t)]
        # timeit.stop('total')
        return outputs[0]


        timeit.start('total')


        ### compute next states
        timeit.start('zero')
        zero_input = torch.autograd.Variable(torch.zeros(input_t.size(0), self._hidden_size).cuda())
        timeit.stop('zero')
        timeit.start('rnn')
        h_tp1s, c_tp1s = zip(*[cell(zero_input, hc_t) for cell in self._rnn_cells])
        timeit.stop('rnn')


        # import IPython; IPython.embed()
        # h_tp1 = torch.autograd.Variable(torch.zeros(input_t.size(0), self._hidden_size).cuda())
        # c_tp1 = torch.autograd.Variable(torch.zeros(input_t.size(0), self._hidden_size).cuda())
        # indices = input_t.data.max(1)[1].squeeze(1)
        # h_tp1[indices] = h_tp1s[indices][indices]
        # c_tp1[indices] = c_tp1s[indices][indices]
        # for index in indices:
        #     h_tp1[index, :] = h_tp1s[index][index, :]
        #     c_tp1[index, :] = c_tp1s[index][index, :]



        # h_tp1 = torch.autograd.Variable(torch.zeros(input_t.size(0), self._hidden_size).cuda())
        # c_tp1 = torch.autograd.Variable(torch.zeros(input_t.size(0), self._hidden_size).cuda())
        # for i, (h, c) in enumerate(zip(h_tp1s, c_tp1s)):
        #     h_tp1 += input_t[:, i].repeat(1, self._hidden_size) * h
        #     c_tp1 += input_t[:, i].repeat(1, self._hidden_size) * c



        # input_t_split = input_t.split(1, dim=1)
        # h_tp1 = sum([mask.repeat(1, self._hidden_size) * h for mask, h in zip(input_t_split, h_tp1s)])
        # c_tp1 = sum([mask.repeat(1, self._hidden_size) * h for mask, h in zip(input_t_split, c_tp1s)])


        timeit.start('combine')
        ### calculate mask
        mask = input_t.unsqueeze(2).repeat(1, 1, self._hidden_size)
        h_tp1s = torch.stack(h_tp1s, 2) # [batch_size, input_size, hidden_size]
        c_tp1s = torch.stack(c_tp1s, 2) # [batch_size, input_size, hidden_size]

        ### use the mask
        h_tp1 = (mask * h_tp1s).sum(1).squeeze(1)
        c_tp1 = (mask * c_tp1s).sum(1).squeeze(1)
        timeit.stop('combine')

        timeit.stop('total')

        return h_tp1, c_tp1
