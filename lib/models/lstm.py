import torch
import torch.nn as nn

class AudioLSTM(nn.Module):

    def __init__(self, input_size=174, hidden=300, num_classes=10):

        super(AudioLSTM, self).__init__()
        self.hidden = hidden
        self.input_size = input_size
        self.num_classes = num_classes

        #self.ln_1 = nn.LayerNorm(self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden, num_layers=2, batch_first=True, dropout=0.2)
        #self.ln_2 = nn.LayerNorm(self.hidden)
        self.linear = nn.Linear(self.hidden, self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.i_defender = None

    def return_hidden_state_memory(self):
        final_list = []
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                final_list.append(m.hidden_states)
                m.hidden_states = []
        return final_list

    def insert_forward_hooks(self, input_shape, cuda=True):
        self.hook_list = []

        def hook(module, input, output):
            module.hidden_states.append(input[0].data) #store the hidden state for each input in the fc layer

        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                layer_hook = m.register_forward_hook(hook)
                self.hook_list.append(layer_hook)
                m.hidden_states = []
        if cuda:
            self.forward(torch.randn(input_shape, dtype=torch.float).unsqueeze(0).cuda()) #add the hooks via a dry pass with random input
        else:
            self.forward(torch.randn(input_shape, dtype=torch.float).unsqueeze(0))
        # clear our the random input
        self.return_hidden_state_memory()

    def remove_hooks(self):
        for hook in self.hook_list:
            hook.remove()

    def update_defender(self, defender):
        self.i_defender = defender
        self.is_attacked_log = []
        self.raw_prob_log = []

    def fetch_attack_log(self):
        return_val = self.is_attacked_log
        self.is_attacked_log = []
        return return_val
    
    def fetch_raw_probs(self):
        return_val = self.raw_prob_log
        self.raw_prob_log = []
        return return_val

    def forward(self, x):
        #x = self.ln_1(x)
        self.lstm.flatten_parameters()
        x, hidden = self.lstm(x)
        #x = self.ln_2(x)
        hidden_state = x[:,-1,:]
        x = self.linear(hidden_state)
        self.relu(x)
        predictions = torch.argmax(x, 1)
        if self.i_defender is not None:
            is_attacked = []
            prob_log = []
            attack_num = 0
            for state, pred in zip(hidden_state, predictions):
                attacked, log_prob = self.i_defender.estimate(state.unsqueeze(0), pred)
                is_attacked.append(attacked)
                prob_log.append(log_prob)
                attack_num += attacked
            self.is_attacked_log.append(is_attacked)
            self.raw_prob_log.append(prob_log)
        return x


def audio_lstm(input_size=174, hidden=300, num_classes=10):
    model = AudioLSTM(input_size, hidden, num_classes)
    return model
        
