import torch
import torch.nn as nn


class CPLSTM(nn.Module):
    def __init__(
            self,
            in_dim,
            rnn_info,
            device,
            bound=False
        ):
        super(CPLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=rnn_info['hidden_size'],
            num_layers=rnn_info['num_layers'],
            device=device
        )
        self.fc = nn.Linear(rnn_info['hidden_size'], 1, device=device)

        self._parameters = {}
        for name, param in self.rnn.named_parameters():
            self._parameters['rnn_' + name] = param
        for name, param in self.fc.named_parameters():
            self._parameters['fc_' + name] = param

        self.h = None
        self.c = None
        self.bound = bound
        self.device = device

    def forward(self, x, lb=None, ub=None):
        if self.h is None:
            if x.ndim == 2:
                self.h = torch.zeros(self.rnn.num_layers, self.rnn.hidden_size).to(self.device)
                self.c = torch.zeros(self.rnn.num_layers, self.rnn.hidden_size).to(self.device)
            elif x.ndim == 3:
                batch_size = x.shape[1]
                self.h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
                self.c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        output, (hn, cn) = self.rnn(x.to(self.device), (self.h, self.c))
        output = self.fc(output)
        if lb is not None and ub is not None and self.bound:
            ub = ub.to(self.device)
            lb = lb.to(self.device)
            if x.ndim == 3:
                assert ub.shape == (x.shape[0], x.shape[1], 1)
                assert lb.shape == (x.shape[0], x.shape[1], 1)
            elif x.ndim == 2:
                assert ub.shape == (x.shape[0], 1)
                assert lb.shape == (x.shape[0], 1)
            assert lb.shape == output.shape
            assert ub.shape == output.shape
            output = (torch.tanh(output) + 1) * (ub - lb) / 2

        self.h = hn
        self.c = cn
        return output
    
    def reset(self):
        self.h = None
        self.c = None


class CPRNN(nn.Module):
    def __init__(
            self,
            in_dim,
            rnn_info,
            device,
            bound=False
        ):
        super(CPRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=rnn_info['hidden_size'],
            num_layers=rnn_info['num_layers'],
            device=device
        )
        self.fc = nn.Linear(rnn_info['hidden_size'], 1, device=device)

        self._parameters = {}
        for name, param in self.rnn.named_parameters():
            self._parameters['rnn_' + name] = param
        for name, param in self.fc.named_parameters():
            self._parameters['fc_' + name] = param
        
        self.h = None
        self.bound = bound
        self.device = device

    def forward(self, x, lb=None, ub=None):
        if self.h is None:
            if x.ndim == 2:
                self.h = torch.zeros(self.rnn.num_layers, self.rnn.hidden_size).to(self.device)
            elif x.ndim == 3:
                batch_size = x.shape[1]
                self.h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        output, hn = self.rnn(x.to(self.device), self.h)
        output = self.fc(output)
        if lb is not None and ub is not None and self.bound:
            ub = ub.to(self.device)
            lb = lb.to(self.device)
            if x.ndim == 3:
                assert ub.shape == (x.shape[0], x.shape[1], 1)
                assert lb.shape == (x.shape[0], x.shape[1], 1)
            elif x.ndim == 2:
                assert ub.shape == (x.shape[0], 1)
                assert lb.shape == (x.shape[0], 1)
            assert lb.shape == output.shape
            assert ub.shape == output.shape
            output = (torch.tanh(output) + 1) * (ub - lb) / 2

        self.h = hn
        return output
    
    def reset(self):
        self.h = None


class CPFFN(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            n_layers,
            device,
            bound=False
        ):
        super(CPFFN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, device=device)
        self.layers = [self.fc1]
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
        self.fc = nn.Linear(hidden_dim, 1, device=device)
        self.layers.append(self.fc)

        self._parameters = {}
        for i, layer in enumerate(self.layers):
            for j, (name, param) in enumerate(layer.named_parameters()):
                self._parameters[f'layer_{i}' + name] = param

        self.h = None
        self.bound = bound
        self.device = device

    def forward(self, x, lb=None, ub=None):
        # x shape is (T, B, N) where T is time, B is batch and N is number of covariates
        assert not torch.isnan(x).any()
        output = x.to(self.device)
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i != len(self.layers) - 1:
                output = torch.relu(output)
        assert not torch.isnan(output).any()

        if lb is not None and ub is not None and self.bound:
            ub = ub.to(self.device)
            lb = lb.to(self.device)
            if x.ndim == 3:
                assert ub.shape == (x.shape[0], x.shape[1], 1)
                assert lb.shape == (x.shape[0], x.shape[1], 1)
            elif x.ndim == 2:
                assert ub.shape == (x.shape[0], 1)
                assert lb.shape == (x.shape[0], 1)
            assert lb.shape == output.shape
            assert ub.shape == output.shape
            output = (torch.tanh(output) + 1) * (ub - lb) / 2
        assert not torch.isnan(output).any()
        return output
    
    def reset(self):
        pass


class CPLinear(nn.Module):
    def __init__(
            self,
            in_dim,
            device="cpu",
            bound=False,
            logtime=False,
            loggap=False,
        ):
        super(CPLinear, self).__init__()

        self._parameters = {}
        self.fc = nn.Linear(in_dim, 1).to(device)
        for name, param in self.fc.named_parameters():
            self._parameters[name] = param
        self.logtime = logtime
        self.loggap = loggap
        self.bound = bound
        self.device = device

    def forward(self, x, lb=None, ub=None):

        x = x.to(self.device)
        if self.logtime:
            x[:2] = torch.log(x[:2])
        output = self.fc(x)
        if self.loggap:
            output = torch.exp(output)
        if lb is not None and ub is not None and self.bound:
            ub = ub.to(self.device)
            lb = lb.to(self.device)
            if x.ndim == 3:
                assert ub.shape == (x.shape[0], x.shape[1], 1)
                assert lb.shape == (x.shape[0], x.shape[1], 1)
            elif x.ndim == 2:
                assert ub.shape == (x.shape[0], 1)
                assert lb.shape == (x.shape[0], 1)
            assert lb.shape == output.shape
            assert ub.shape == output.shape
            output = (torch.tanh(output) + 1) * (ub - lb) / 2
        return output

    def reset(self):
        pass
