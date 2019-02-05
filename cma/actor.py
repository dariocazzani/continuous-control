import numpy as np

class Actor(object):
    def __init__(self, state_size, num_actions, num_hidden=0):
        self.num_hidden = num_hidden
        self.num_inputs = state_size
        self.num_outputs = num_actions
        if self.num_hidden == 0:
            self.num_params = self.num_inputs * self.num_outputs + self.num_outputs
            
        else:
            self.num_weights_in = self.num_inputs*self.num_hidden
            self.num_bias_in = self.num_hidden
            self.num_weights_hidden = self.num_hidden * self.num_outputs
            self.num_bias_hidden = self.num_outputs

            self.num_params = self.num_weights_in + self.num_bias_in + self.num_weights_hidden + self.num_bias_hidden

    def _get_weights_bias_no_hidden(self, params):
        """
            params: list of lenght "num_agents" of parameters for the actor
        """
        agents_weights = []
        agents_bias = []
        
        for p in params:
            weights = p[:self.num_params - self.num_outputs]
            bias = p[-self.num_outputs:]
            weights = np.reshape(weights, [self.num_inputs, self.num_outputs])
            agents_weights.append(weights)
            agents_bias.append(bias)
        return agents_weights, agents_bias

        
    def _get_weights_bias_hidden(self, params):
        """
            params: list of lenght "num_agents" of parameters for the actor
        """
        agents_weights_in = []
        agents_weights_hidden = []
        agents_bias_in = []
        agents_bias_hidden = []

        for p in params:
            p = list(p)
            weights_in = p[:self.num_weights_in]
            del p[:self.num_weights_in]
            weights_in = np.reshape(weights_in, [self.num_inputs, self.num_hidden])
            agents_weights_in.append(weights_in)

            bias_in = p[:self.num_bias_in]
            del p[:self.num_bias_in]
            agents_bias_in.append(bias_in)
            
            weights_hidden = p[:self.num_weights_hidden]
            del p[:self.num_weights_hidden]
            weights_hidden = np.reshape(weights_hidden, [self.num_hidden, self.num_outputs])
            agents_weights_hidden.append(weights_hidden)

            bias_hidden = p[:self.num_bias_hidden]
            del p[:self.num_bias_hidden]
            agents_bias_hidden.append(bias_hidden)
            
        return agents_weights_in, agents_bias_in, agents_weights_hidden, agents_bias_hidden
    
    def get_num_params(self):
        return self.num_params

    def decide_actions(self, observation, params):
        predictions = []
        if self.num_hidden == 0:
            agents_weights, agents_bias = self._get_weights_bias_no_hidden(params)
            for idx, w in enumerate(agents_weights):
                prediction = np.matmul(np.squeeze(observation[idx]), w) + agents_bias[idx]
                prediction = np.tanh(prediction)
                predictions.append(prediction)
        else:
            agents_weights_in, agents_bias_in, agents_weights_hidden, agents_bias_hidden = self._get_weights_bias_hidden(params)
            for idx, w in enumerate(agents_weights_in):
                hidden_layer = np.tanh(np.matmul(np.squeeze(observation[idx]), w) + agents_bias_in[idx])
                prediction = np.matmul(hidden_layer, agents_weights_hidden[idx]) + agents_bias_hidden[idx]
                prediction = np.tanh(prediction)
                predictions.append(prediction)

        return predictions