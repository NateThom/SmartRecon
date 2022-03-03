from torch import nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_nodes_per_layer, activation='ReLU', dropout=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)

        self.layers.add_module(f'fc_{0}', nn.Linear(in_features, num_nodes_per_layer))
        self.layers.add_module(f'{activation}_{0}', act_fn())
        self.layers.add_module(f'dropout_{0}', nn.Dropout(dropout))

        for i in range(num_layers):
            self.layers.add_module(f'fc_{i+1}', nn.Linear(num_nodes_per_layer, num_nodes_per_layer))
            self.layers.add_module(f'{activation}_{i+1}', act_fn())
            self.layers.add_module(f'dropout_{i+1}', nn.Dropout(dropout))

        self.layers.add_module(f'fc_{num_layers+1}', nn.Linear(num_nodes_per_layer, out_features))
        # self.layers.add_module(f'{activation}_{num_layers+1}', act_fn())
        # self.layers.add_module(f'dropout_{num_layers+1}', nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)
