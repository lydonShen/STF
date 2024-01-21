import numpy as np
import random
from eod_replay import getdata

def create_dataset(num_steps, data_dir_prefix,log_dir_prefix):
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []
    data_iterator1=getdata(log_dir_prefix+"transition6.txt")
    data_iterator2=getdata(log_dir_prefix+"transition7.txt")
    data_iterator3=getdata(log_dir_prefix+"transition9.txt")
    # data_iterator4=getdata(log_dir_prefix+"transition13.txt")
    # data_iterator5=getdata(log_dir_prefix+"transition11.txt")
    # data_iterator6=getdata(log_dir_prefix+"transition10.txt")
    # data_iterator7 = getdata(log_dir_prefix + "transition8.txt")
    # with open("") as file:
    #     lines = file.readlines()
    #     # print(len(lines))
    # benefitdict = {}
    # for i in lines:
    #     replace = i.split(",")
    #     benefitdict[replace[0]] = float(replace[1])
    # lossdict= benefitdict

    num_trajectories_1 = 0
    num_trajectories_2 = 0
    num_trajectories_3 = 0
    # num_trajectories_4 = 0
    # num_trajectories_5 = 0
    # num_trajectories_6 = 0
    # num_trajectories_7 = 0

    while len(obss) < num_steps:
        terminal = False
        random_int = random.randint(1,3)
        if random_int==1:
            if num_trajectories_1<3000:
                while not terminal:
                    # obss
                    observationname, ret,ac,terminal = next(data_iterator1)

                    #print (terminal)
                    statesname = data_dir_prefix+str(observationname)+".npy"
                    states = np.load(statesname,allow_pickle=True)

                    #print(states[0].shape)
                    states = states.transpose((2, 0, 1)) # (100, 180, 64) --> (256, 100, 180)
                    obss += [states]

                    # ac
                    actions += [ac]

                    # ret
                    stepwise_returns += [ret]
                    returns[-1] += ret

                done_idxs += [len(obss)]
                returns += [0]
                num_trajectories_1 = num_trajectories_1+1

        elif random_int==2:
            if num_trajectories_2<3000:
                while not terminal:
                    observationname, ret,ac,terminal = next(data_iterator2)
                    statesname = data_dir_prefix+str(observationname)+".npy"
                    states = np.load(statesname,allow_pickle=True)
                    states = states.transpose((2, 0, 1))
                    obss += [states]
                    actions += [ac]
                    stepwise_returns += [ret]
                    returns[-1] += ret
                done_idxs += [len(obss)]
                returns += [0]
                num_trajectories_2 = num_trajectories_2+1

        elif random_int==3:
            if num_trajectories_3<3000:
                while not terminal:
                    observationname, ret,ac,terminal = next(data_iterator3)
                    statesname = data_dir_prefix+str(observationname)+".npy"
                    states = np.load(statesname,allow_pickle=True)
                    states = states.transpose((2, 0, 1)) # (1, 100, 180, 256) --> (256, 100, 180)
                    obss += [states]
                    actions += [ac]
                    stepwise_returns += [ret]
                    returns[-1] += ret
                done_idxs += [len(obss)]
                returns += [0]
                num_trajectories_3 = num_trajectories_3 + 1

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # return-to-go 
    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
            #print(rtg[j])
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))
    
    return obss, actions, returns, done_idxs, rtg, timesteps

# if __name__ == '__main__':
#     data = create_dataset()
#     #1w 40G
