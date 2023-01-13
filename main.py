from gameenv import GameEnv
from ppo import ShooterAgent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = GameEnv()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    shooterAgent = ShooterAgent(n_actions=env.shooter_action_space.n, batch_size=batch_size, alpha=alpha,
                                n_epochs=n_epochs, input_dims=env.observation_space.shape)
    collectorAgent = ShooterAgent(n_actions=env.collector_action_space.n, batch_size=batch_size, alpha=alpha,
                                  n_epochs=n_epochs, input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = 10
    score_history = []

    learn_iters = 0
    ag_score = 0
    n_steps = 0

    env.run_game_loop()

    for i in range(n_games):
        observation = env.reset_game()
        done = False
        score = 0
        while not done:
            action, prob, val = shooterAgent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            shooterAgent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                shooterAgent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            shooterAgent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)


running = True
while running:
    break
