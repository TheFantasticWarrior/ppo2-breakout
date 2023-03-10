# needs a folder named save in same directory
import numpy as np
from multiprocessing import Process, Pipe
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
import gym
import cv2
import gc
import csv
from collections import deque
from gym import spaces


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#config
render=1 #gym 0.13.1 is easier to render because later versions seem to limit fps
load=1
nenvs=16
nsteps=128
nminibatch=4
nepoch=4

gamma=0.99
lam=0.95
loops=int(1e+7//nsteps//nenvs)
seed=None


#use dictionary and local to log because this looks simpler imo
losses=["timestep","ent", "vf_loss", "gradnorm",\
    "re","approxkl","clipfrac","ev","mean_rew"]


def main():
    if load:
        m=tf.saved_model.load('save/model')
    else:
        m=model(nenvs,action_space=4,lr=2.5e-4)
    remotes, work_remotes = zip(*[Pipe() for _ in range(nenvs)])
    processes=[Process(target=gather_data,args=(work_remote, remote,i==0,render)) for i,(work_remote,remote) in enumerate(zip(work_remotes,remotes))]
    for p in processes:
        p.daemon=True
        p.start()
    wos=remotes[0].recv()
    venv=VecFrameStack(4,wos,remotes)
    obs=venv.reset()
    try:
        for loop in range(loops):
            ep_rew=[]
            ep_len=[]
            for i in range(nsteps):
                acs,nlps,pred_ext=m.step(tf.constant(obs))
                for remote,action in zip(remotes,acs.numpy()):
                    remote.send(('step',action))
                buf_obs[:,i]=obs.copy()
                buf_nlps[:,i]=nlps.numpy()
                buf_acs[:,i]=acs

                buf_pred_ext[:,i]=pred_ext.numpy().flatten()
                obs[:],rew,done,info=venv.step_wait()
                #print(rew)
                ep_rew+=[i["rew"] for i in info if i is not None]
                
                ep_len+=[i["len"] for i in info if i is not None]
                buf_rews_ext[:,i]=rew
                dones[:,i]=done
            timestep=(loop+1)*nenvs*nsteps
            pred_ext_last=m.val(tf.constant(obs)).numpy().flatten()
            mean_rew=mean(ep_rew)
            print(f"loop {loop:4} end, mean episode reward {mean_rew:5.1f},"+
                    f"mean episode length {mean(ep_len):.1f}, {dones.sum():.0f} deaths")
            
            lastgaelam = 0
            for t in reversed(range(nsteps)):
                nextvals = buf_pred_ext[:, t + 1] if t + 1 < nsteps else pred_ext_last
                nextnotdone = 1 - dones[:,t]

                delta = buf_rews_ext[:, t] + gamma * nextvals * nextnotdone - buf_pred_ext[:, t]
                buf_advs_ext[:, t] = lastgaelam = delta + gamma * lam * nextnotdone * lastgaelam
            bufs_rets_ext = buf_advs_ext +buf_pred_ext
            
            ev=explained_variance(buf_pred_ext.swapaxes(0, 1).flatten(),bufs_rets_ext.swapaxes(0, 1).flatten())
            print(f"{ev=}")
            for i in range(nepoch):
                order=np.arange(nenvs*nsteps)
                np.random.shuffle(order)
                start=0
                for j in range(nminibatch):
                    sli=order[start:start+samplesperbatch]
                    start+samplesperbatch
                    
                    pactions,nlp,adv,rets_ext,old_pred_ext=map(lambda x:tf.constant(x.swapaxes(0, 1).flatten()[sli]),\
                        (buf_acs,buf_nlps,buf_advs_ext,bufs_rets_ext,buf_pred_ext))
                    states=tf.constant(buf_obs.swapaxes(0, 1).reshape(-1,84,84,4)[sli])
                        
                    #if j==0:breakpoint()
                    adv=(adv-tf.reduce_mean(adv))/(tf.math.reduce_std(adv)+1e-8)
                    re=np.mean(rets_ext) #for logging
                
                    vf_loss,ent,approxkl,clipfrac,gradnorm=m.train(pactions,states,nlp,adv,rets_ext,old_pred_ext)
                    
                    for x in losses:
                        D[x]=D[x]+[float(locals()[x])]
            gc.collect()
            if (loop+1)%(20)==0:
                with open('save/log.csv', 'w') as f:
                    w = csv.writer(f)
                    w.writerow(D.keys())
                    w.writerows(zip(*(running_mean(x,nminibatch*nepoch) for x in D.values())))
            if ((loop+1)*nenvs*nsteps)%2048000==0:
                tf.saved_model.save(m, 'save/model')
        
        tf.saved_model.save(m, 'save/model')
    except KeyboardInterrupt:
        x=input("save?[Y/n]")
        if x in ("Y",'y',""):
            tf.saved_model.save(m, 'save/model')
#todo: apparently keras model class is better but I'm still figuring things out
class model(tf.Module):
    def __init__(self, batch_size,action_space,cliprange=0.1,lr=2.5e-4,
                 vf_coef=0.5,ent_coef=0.01,total_timesteps=1e7):
        super(model,self).__init__(name="PPO2")
        self.batch_size=batch_size
        self.cliprange=cliprange
        self.lr=lr
        self.vf_coef=vf_coef
        self.ent_coef=ent_coef

        self.nupdates=int(total_timesteps/nsteps//nenvs)
        self.lrdiscount=self.lr/self.nupdates
        net=network()
        with tf.name_scope("net"):
            
            self.pol_net=net
            self.val_net=net
            self.policy=layers.Dense(action_space,kernel_initializer=tf.keras.initializers.Orthogonal(0.01),name="action")
            self.value=layers.Dense(1,kernel_initializer=tf.keras.initializers.Orthogonal(1),name="value")
            self.policy.build(self.pol_net.output_shape)
            self.value.build(self.val_net.output_shape)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr,epsilon=1e-5)
    @tf.function(jit_compile=True,reduce_retracing=True,
    input_signature=[tf.TensorSpec([None,84,84,4], tf.uint8)])
    def step(self,obs):
        pol_latent=self.pol_net(obs)
        pd=self.policy(pol_latent)
        u=tf.random.uniform(tf.shape(pd))
        acs=tf.argmax(pd-tf.math.log(-tf.math.log(u)),axis=-1)
        nlps=tf.nn.sparse_softmax_cross_entropy_with_logits(acs,pd)

        val_latent=self.val_net(obs)
        pred_ext=self.value(val_latent)
        return acs,nlps,pred_ext
    @tf.function(jit_compile=True,reduce_retracing=True,
    input_signature=[tf.TensorSpec([None,84,84,4], tf.uint8)])
    def val(self,obs):
        return tf.squeeze(self.value(self.val_net(obs)))

    @tf.function(jit_compile=True,reduce_retracing=True,
    input_signature=[tf.TensorSpec([None], tf.int32),
    tf.TensorSpec([None,84,84,4], tf.float32),tf.TensorSpec([None], tf.float32),
    tf.TensorSpec([None], tf.float32),
    tf.TensorSpec([None], tf.float32),
    tf.TensorSpec([None], tf.float32)])
    def train(self,pactions,states,nlp,adv,rets_ext,old_pred_ext):
        with tf.GradientTape() as tape:
            
            pol_latent=self.pol_net(states)
            self.pd=self.policy(pol_latent)
            
            val_latent=self.val_net(states)
            v=tf.squeeze(self.value(val_latent))
            
            pred_ext_clipped=old_pred_ext+tf.clip_by_value(v - old_pred_ext,-self.cliprange,self.cliprange)
            
            le1=tf.square(pred_ext_clipped- rets_ext)
            le2=tf.square(v - rets_ext)
            vf_loss_ext = 0.5*tf.reduce_mean(tf.maximum(le1,le2))
            vf_loss = self.vf_coef*vf_loss_ext
            
            ent=tf.reduce_mean(self.entropy())

            neglogpac=tf.nn.sparse_softmax_cross_entropy_with_logits(pactions,self.pd)
            ratio = tf.exp(nlp-neglogpac)
            negadv = - adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            ent_loss =  (-self.ent_coef) * ent
            
            loss = pg_loss + ent_loss + vf_loss 
        grads = tape.gradient(loss, self.trainable_variables)
        grads,gradnorm=tf.clip_by_global_norm(grads,0.5)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                    
        approxkl = .5 * tf.reduce_mean(tf.square(nlp-neglogpac))
        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), self.cliprange),tf.float32))
            
        self.lr-=self.lrdiscount
        self.optimizer.learning_rate.assign(self.lr)
        return vf_loss_ext,ent,approxkl,clipfrac,gradnorm

    
    def entropy(self):
        logits=self.pd
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
def network():
    inputs=layers.Input(shape=(84,84,4))
    
    h = tf.cast(inputs, tf.float32) / 255.
    def create_cnnblock(layer,nf,fs,s):
        layer2 = layers.Conv2D(nf, fs,strides=s,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))(layer)
        #layer2=layers.BatchNormalization()(layer2)
        layer2=layers.LeakyReLU()(layer2)
    
        return layer2
    
    
    layer1=create_cnnblock(h,32,8,4)
    layer2=create_cnnblock(layer1,64,4,2)
    layer3=create_cnnblock(layer2,64,3,1)
    layer3=layers.Flatten()(layer3)
    layer5 = layers.Dense(512,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),activation='relu')(layer3)
    #layer6 = layers.Dense(448,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))(layer5)

    #layer71 = layers.Dense(32,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),activation='relu')(layer5)
    #layer72 = layers.Dense(64,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),activation='relu')(layer5)
    #layer71 = layers.Dense(32,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),activation='relu')(layer71)
    #layer72 = layers.Dense(64,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),activation='relu')(layer72)
    
    
    return Model(inputs=inputs,outputs=layer5)
def explained_variance(ypred,y):
        """
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]
        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero
        """
        try:
            assert y.ndim == 1 and ypred.ndim == 1
        except:
            assert y.shape==ypred.shape
            y=y.flatten()
            ypred=ypred.flatten()
            assert y.ndim == 1 and ypred.ndim == 1
        vary = np.var(y)
        return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
if seed:
    tf.random.set_seed(seed)
    np.random.seed(seed)
D={key:[] for key in losses}
D["gradnorm"]=[]
samplesperbatch=nenvs*nsteps//nminibatch
buf_obs=np.zeros((nenvs, nsteps,84,84,4), np.float32)

#buf_pd=np.zeros((nenvs, nsteps,9), np.float32)
buf_nlps=np.zeros((nenvs, nsteps),np.float32)
buf_acs=np.zeros((nenvs, nsteps),np.int32)

buf_pred_ext=np.zeros((nenvs, nsteps),np.float32)
buf_advs_ext=np.zeros((nenvs, nsteps), np.float32)
buf_rews_ext=np.zeros((nenvs, nsteps),np.float32)
buf_advs_ext=np.zeros((nenvs, nsteps), np.float32)
dones=np.zeros((nenvs, nsteps), np.float32)
def mean(x):
    if len(x):
        return np.mean(x)
    else:
        return np.nan
def running_mean(x, N):
    l=[]
    for i in range(0,len(x),N):
        l.append(np.mean(x[i:i+N]))
    return l

def gather_data(remote,parent_remote,first,render):
    parent_remote.close()
    env=gym.make("BreakoutNoFrameskip-v4")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env.seed(seed)
    env = WarpFrame(env)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    if first:
        remote.send(env.observation_space)
    try:
        while True:
            cmd,action=remote.recv()
            if cmd=='step':
                
                x,reward,done,_=env.step(action)
                
                if render and first:
                    env.render()
                if done:
                    x,info=env.reset()
                else:
                    info=None
                remote.send((x,reward,done,info))
            elif cmd=='reset':
                x,_=env.reset()
                remote.send(x)
            elif cmd=='close':
                env.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=10):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 1
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs,_ = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)      

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        Modified to log episode rewards and length
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        
        self.l=0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        
        self.total_reward+=reward*(reward>0)
        self.l+=1
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, None

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
            if self.l:
                info={"rew":self.total_reward,"len":self.l}
            else:
                info=None
            self.total_reward=0
            self.l=0
        else:
            info=None
            # original:no-op step to advance from terminal/lost life state
            # modified to do "FIRE" action instead so the game would continue
            obs, _, _, _ = self.env.step(1)
        self.lives = self.env.unwrapped.ale.lives()
        return obs,info

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class VecFrameStack():
    def __init__(self, nstack,wos,remotes):
        low = np.repeat(wos.low, nstack, axis=-1)
        self.stackedobs = np.zeros((nenvs,) + low.shape, low.dtype)
        self.remotes=remotes

    def step_wait(self):
        ob,rews,news,infos=zip(*[remote.recv() for remote in self.remotes])
        try:
            obs=np.stack(ob)
        except:
            breakpoint()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, np.stack(rews), np.stack(news), infos

    def reset(self):
        
        for remote in self.remotes:
            remote.send(('reset',None))
        obs= [remote.recv() for remote in self.remotes]
        obs= np.stack(obs)
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs


if __name__=="__main__":
    main()
    