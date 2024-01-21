def getdata(dir):
    with open(dir) as file:
        lines = file.readlines()
        #print(len(lines))

    #segmented and convert formats
    def splitlist(splist):
        context = []
        for i in splist:
            #print(i.replace("[","").replace("]",""))
            replace = i.replace("[","").replace("]","").replace("'", "").replace("\n","")
            #print(replace.split(","))
            for j in replace.split(","):
                #print(int(j))
                context.append(int(j))
        return context

    def splitlist2(splist):
        context = []
        for i in splist:
            replace = i.replace("[","").replace("]","").replace("'", "").replace("\n","")
            #print(replace.split(","))
            for j in replace.split(","):
                #print(float(j))
                context.append(float(j))
        return context

    def splitlist3(splist):
        context = []
        for i in splist:
            replace = i.replace("[","").replace("]","").replace("'", "").replace("\n","")
            #print(replace.split(","))

            for j in replace.split(","):
                #print(float(j))
                context.append(int(j))
            context.append(9)
        return context     

    def splitlist4(splist):
        context = []
        for i in splist:
            replace = i.replace("[","").replace("]","").replace("'", "").replace("\n","")
            #print(replace.split(", "))
            for j in replace.split(","):
                if j != " "+str(True):
                    context.append(bool(False))
                else:
                    context.append(bool(True))
        return context

    def splitlist5(splist):
        context = []
        for i in splist:
            replace = i.replace("[","").replace("]","").replace("'", "").replace("\n","")
            for j in replace.split(","):
                #print(float(j))
                context.append(float(j))
            context.append(0)
        return context

    stateslist = lines[::5]
    returnslist = lines[1::5]
    actionslist = lines[2::5]
    terminallist = lines[3::5]
    rewardlist = lines[4::5]

    states = splitlist(stateslist)
    print(len(states))
    print(states[0])

    returns = splitlist2(returnslist)
    print(len(returns))
    print(returns[0])


    actions = splitlist3(actionslist)
    print(len(actions))
    print(actions[0])

    terminals = splitlist4(terminallist)
    print(len(terminals))
    print((terminals[0]))

    reward = splitlist5(rewardlist)
    print(len(reward))
    print((reward[0]))

    data_iterator = list(zip(states, reward, actions, terminals))
    return data_iterator

    # for _ in len(states):
    #     observationname,ret,ac,term= next(data_iterator)
    #     imagename = str(observationname)+".npy"
    #     observation = np.load(imagename)
    #     action = np.array(ac)
    #     reward = np.array(ret)
    #     terminal = np.array(terminals)









