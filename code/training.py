__author__ = 'Guillaume'

import dqn
import deepnets

epsilon = 0.8
gamma= 0.99
batch_size = 16
target_patience = 1
buffer_size = 50*500

if False:
    rec_depth = 50
    # Environment
    environment = dqn.GridWorld(7, 7, 5, 3, 50, gamma=gamma, centered=True, p=0.5)
    # A neural net
    neuralnet = deepnets.stateful_RCNN_7x7((batch_size, 3, 7, 7), 0.01, adam=False)
    neuralnet_copy = deepnets.stateful_RCNN_7x7((batch_size, 3, 7, 7), 0.01, adam=False)
    targetnet = deepnets.stateful_RCNN_7x7((batch_size, 3, 7, 7), 0.01, adam=False)
    # ExperienceReplay
    experience_replay = dqn.TemporalExperienceReplay(rec_depth, environment.return_state().shape, buffer_size)
    # Agent
    agent = dqn.StatefulRDQNAgent(environment, neuralnet, neuralnet_copy, experience_replay, batch_size, target_net=targetnet,
                                  flatten=False, target_net_patience=target_patience, epsilon=epsilon, gamma=gamma)
    name = "RDQN_p=0.5_sgd"
else:
    rec_depth = 50
    # Environment
    environment = dqn.GridWorld(7, 7, 5, 3, 50, gamma=gamma, centered=True, p=0.5)
    # A neural net
    neuralnet = deepnets.stateful_RMCNN_7x7((batch_size, 3, 7, 7), (batch_size, 10, 3, 7, 7), 0.01, adam=True)
    neuralnet_copy = deepnets.stateful_RMCNN_7x7((batch_size, 3, 7, 7), (batch_size, 10, 3, 7, 7), 0.01, adam=True)
    targetnet = deepnets.stateful_RMCNN_7x7((batch_size, 3, 7, 7), (batch_size, 10, 3, 7, 7), 0.01, adam=True)
    # ExperienceReplay
    experience_replay = dqn.TemporalExperienceReplay(rec_depth, environment.return_state().shape, buffer_size, memory=10)
    # Agent
    agent = dqn.StatefulDRMQNAgent(environment, neuralnet, neuralnet_copy, experience_replay, batch_size, target_net=targetnet,
                                   flatten=False, target_net_patience=target_patience, epsilon=epsilon, gamma=gamma)
    name = "DRMQN_p=0.5_adam"

# Train for 100 epochs of 1500 updates
agent.fit(3000, 100, max_q_size=10, savename="RL/experiment_03/weights_%s.npy"%name)
# Save learning curve
agent.save_curve("RL/experiment_03/curve_%s.npy"%name)