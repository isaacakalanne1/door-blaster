from gameenv import GameEnv
from ppo import Agent
from helper import plot
import numpy as np

if __name__ == '__main__':
    env = GameEnv()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    shooterAgent = Agent(n_actions=9, input_dims=env.get_shooter_state().shape, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    collectorAgent = Agent(n_actions=5, input_dims=env.get_collector_state().shape, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    n_games = 300

    best_score = 10
    score_history = []

    plot_scores = []
    plot_mean_scores = []

    learn_iters = 0
    ag_score = 0
    n_steps = 0

    for i in range(n_games):
        env.reset_game()
        s_observation = env.get_shooter_state()
        c_observation = env.get_collector_state()
        done = False
        score = 0
        while not done:
            s_action, s_prob, s_val = shooterAgent.choose_action(s_observation)
            c_action, c_prob, c_val = collectorAgent.choose_action(c_observation)

            ((s_observation_, s_reward, s_done),
             (c_observation_, c_reward, c_done)) = env.take_game_step(s_action, c_action)

            if s_done:
                done = s_done

            if c_done:
                done = c_done

            n_steps += 1
            score += s_reward + c_reward

            shooterAgent.remember(s_observation, s_action, s_prob, s_val, s_reward, done)
            collectorAgent.remember(c_observation, c_action, c_prob, c_val, c_reward, done)

            if n_steps % N == 0:
                shooterAgent.learn()
                collectorAgent.learn()
                learn_iters += 1
            s_observation = s_observation_
            c_observation = c_observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            shooterAgent.save_models()
            collectorAgent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps,
              'learning_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))]
        plot_scores.append(score)
        plot_mean_scores.append(avg_score)
        plot(plot_scores, plot_mean_scores)
