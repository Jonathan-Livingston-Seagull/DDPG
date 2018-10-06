import gym, roboschool
import random
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()


print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


from ddpg_agent_tf import Agent



ENV_NAME = 'RoboschoolInvertedPendulum-v1'
env = gym.make(ENV_NAME)
env.seed(2)
agent = Agent(state_size=5, action_size=1, random_seed=2)


def ddpg(n_episodes=10000, max_t=3000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        #agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act([state])[0]
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                #print(t)
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.max(scores)), end="")
        #torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        #torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.max(scores)))



    return scores


with tf.device("/gpu:0"):
    scores = ddpg()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()