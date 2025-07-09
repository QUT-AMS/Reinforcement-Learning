
Reinforcement Learning (RL)  ==>  Area of ML concerned with how softw. AGENTS ought to take action in an env. to max. otion of cumulat. reward

OR teaching AGENTS to behave in env. by telling it how good it's doing

  TRAINING MODEL: =>   DEEP Q LEARNING approach & extends REINFORCEMENT LEARNING ==>  PREDICTS ACTIONS in DEEP NEURAL NETWORK BY REWARDING QUALITY OF ACTION

  PySnake

  3 Main Parts:   1. GAME  2. AGENT 3 .MODEL

  [1: GAME  ] =>  2D-game

  [2: MODEL ] =>  PyTorch:Linear_QNet DQN => Predicts ACTION

  [3: AGENT ] =>  Controls  GAME(S) & MODEL]

  TRAINING:    (exploring --> exploit)

  game.get_state(game) ~~> [STATE]     ~~>  game.get_move([STATE]) ~~>  [ACTION]   model.predict([ACTION])  ~~> [REWARD], [GAMEOVER], [SCORE]  ~~> game.play_step()
