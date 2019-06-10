import retro

# print(retro.data.list_games())


# environment = retro.make(game='SonicTheHedgehog-Genesis')
environment = retro.make(game='Airstriker-Genesis')


with environment:
    observation = environment.reset()
    while True:
        environment.render()
        action = environment.action_space.sample()
        observation, reward, done, info = environment.step(action)
        if done:
            observation = environment.reset()
