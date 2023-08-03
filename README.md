# PPO2-Breakout implementation
## Background/About
My implementation of PPO2 in tf2, does get around 400 score sometimes from around 8m steps, but low scores are still low  
Not an improvement or useful tool but I made it more understandable <sup>for myself</sup> by removing things I don't understand and hope it doesn't break, and ordered functions by time used in same file  
Welcome to get inspired from the code, just don't get offended by my bad practices and writings.   

Originally tried to make [RND](https://openai.com/research/reinforcement-learning-with-prediction-based-rewards) in custom environment but failed, so trying in an easier environment with simpler model instead, which caused some weird variable names, did try to make the code less ugly by using class. Wrappers are copy pasted instead of imported because `gym` version changing breaks them and I modified a bit

## bugs fixed / things learned
for those that are trying to implement ppo2 themselves

errors:
- depending on `gym` version there can be 1 or 2 return from `env.reset` and 4 or 5 return from `env.step`, WILL break the code(sometimes gives cv2 error) 

score:
- recorded episode scores wrong
- recorded episode scores wrong ***AGAIN*** (should've simply used `Monitor`)

explained variance:
- make sure to `tf.squeeze` the value prediction, that's what fixed my ev
- make sure you pass in 1d for explained variance, the 2d function returns 2d variance which is hard to log, and make sure you know what `tf.squeeze` does and that it doesn't shrink dimentions if the dimension is larger than 1

multiprocessing:
- for `multiprocessing` send numpy/python values as action instead of tf object
- `VecFrameStack` isn't related to implementing parallel envs, just frame stack but modified to work in parallel, the `SubprocVecEnv` does the job <sup>which is messily separated in main in my code for some reason</sup>

other:
- with `tf.function` decorators you should send tensor instead of numpy to the function, apparently faster
- pass tf.TensorSpec to `tf.function` decorators or else you can't change batch size
- idk if some changes like swap axes before flattening or if using `obs[:]...=env.step` and `buf_obs=obs.copy` help
- `gc` (garbage collect) was for memory leak in my custom environment but [seems to help tensorflow too](https://github.com/keras-team/keras/issues/16019)
- [csv plotting online](https://www.csvplot.com/) is more useful than saving tons of image in `matplotlib` <sup>imo</sup>

unrelated:
- I learned docker, wsl and x server on top of ML stuff
- `baselines` and `gym` are complicated and break often with versions I hate it, but OpenAI's AI are so cool

## Requirements
- needs a folder named `save` in same directory<sup>unless you modify the saving parts of the code</sup>  

- tested in `gym` 0.13.1 <sup>because pip decided to downgrade `gym` for no reason but this works well, other versions I used seems to break rendering (`env.make(...,rendermode="human")` limits fps which slows down training and making only one render is complicated)  </sup>	

- a backup brain <sup>reading my code will probably break your brain</sup>	

## Usage
`ppo.py` for training, `play.py` to show results after training if you set render off or unfortunately use newer `gym` package
