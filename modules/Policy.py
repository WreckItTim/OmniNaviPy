from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Utils
import numpy as np
import torch as th


class DQN(th.nn.Module):
    def __init__(self, cnn_extractor, linear_extractor, vec_extractor, q_net):
        super(DQN, self).__init__()
        self.cnn_extractor = cnn_extractor
        self.linear_extractor = linear_extractor
        self.vec_extractor = vec_extractor
        self.q_net = q_net
    
    def forward(self, img, vec):
        img = self.cnn_extractor(img)
        img = self.linear_extractor(img)
        vec = self.vec_extractor(vec)
        cat = th.cat([img, vec], dim=1)
        q_values = self.q_net(cat)
        return q_values


# this is purely an abstract class to define the predict() function that all policies should have
class Policy(Component.Component):
    def predict(self, observations, episode=None):
        # this function should be implemented by any policy subclass
        # it should take in observations and output an action or action values
        raise NotImplementedError('predict() function not implemented for this policy')

class DQNPolicy(Policy):
    def __init__(self, pytorch_model, device='cuda'):
        self.pytorch_model = pytorch_model.to(device)
        self.device = device

    # inputs observations and otuputs the action_idx of the action with highest q_value
    def predict(self, observations, episode=None):
        img = th.tensor(np.expand_dims(observations['img'].astype(np.float32)/255, axis=0), device=self.device)
        vec = th.tensor(np.expand_dims(observations['vec'].astype(np.float32), axis=0), device=self.device)
        q_values = self.pytorch_model(img, vec)
        action_idx = int(th.argmax(q_values).detach().cpu().numpy())
        return action_idx

def read_dqn_policy(pytorch_model_path, device='cuda'):
    pytorch_model = Utils.pickle_read(pytorch_model_path)
    model = DQNPolicy(pytorch_model, device)
    return model


# def read_sb3_model(model_path, device='cuda'):
#     from OmniNaviPy.modules import SB3Wrapper
#     from stable_baselines3 import DQN
#     model = DQN.load(model_path, device=device)
#     model = SB3Wrapper.ModelSB3(model)
#     return model