import math

def logistic(input):
    return 1/(1+(math.exp(-input)))

def derivative(input):
    return logistic(input)*(1-logistic(input))

def quest10():
    summa = []
    summaExp = []
    w1, w2, b = 1, -1.5, -0.5
    for mask in range(8):
        v1,v2,h = mask&1, int(float(mask&2)/2), int(float(mask&4)/4.0)
        energy = -w1*v1*h - w2*v2*h - b*h
        summa.append(energy)
        summaExp.append(math.exp(-energy))
        print('v1 = %s, v2 = %s, h=%s, energy = %s, exp = %s' % (v1, v2, h, energy, math.exp(-energy)))
    normSumma = [float(i)/sum(summa) for i in summa]
    normSummaExp =  [float(i)/sum(summaExp) for i in summaExp]
    print('summa = %s, expSumma = %s' % (normSumma, normSummaExp))
    print(normSummaExp[6]/(normSummaExp[2]+normSummaExp[6]))

def quest11():
    hinput = 1.0986123
    houtput = logistic(hinput)
    y = 4*houtput
    error = ((y-5)**2)/2
    print('quest11 ', y, error)
    return y

def quest12():
    w1 = 1.0986123
    w2 = 4
    y = quest11()
    print('quest12 ', (y - 5)*w2*derivative(w1))
    return (y - 5)*w2*derivative(w1)

quest10()
quest11()
quest12()