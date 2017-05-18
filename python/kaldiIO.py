import numpy as np
from warnings import warn

def str2num(s):
    floats = [float(i) for i in s.split()]
    floats = np.asarray(floats, dtype=np.float32)
    return floats

def parseComponent(lines):
    #parse component and return a dict and number of "consumed" lines
    if lines[0][0:15] != '<ComponentName>':
        raise('Wrong format - <ComponentName> missing - must be the Kaldi Nnet3 txt format')

    node = {}
    shift = 1
    s1 = lines[0][16:].strip().split()
    node['ComponentName'] = s1[0]
    node['ComponentType'] = s1[1][1:-1]

    #switch node['ComponentType']
    if node['ComponentType'] == 'FixedAffineComponent' or node['ComponentType'] == 'NaturalGradientAffineComponent':
        n = 1
        while lines[n][0] != '<':
            n = n + 1
        dim = len(str2num(lines[1]))
        node['LinearParams'] = np.zeros([n-1, dim], dtype=np.float32)
        for k in range(1, n-2):
            node['LinearParams'][k-1, :] = str2num(lines[k])
        node['LinearParams'][-1, :] = str2num(lines[n-1][0:-2])
        if lines[n][0:12] != '<BiasParams>':
            raise('Wrong format - <BiasParams> missing in <FixedAffineComponent> - must be the Kaldi Nnet3 txt format')
        node['BiasParams'] = str2num(lines[n][15:-2])
        shift = n + 2

    elif node['ComponentType'] == 'PnormComponent':
        node['InputDim'] = int(s1[3])
        node['OutputDim'] = int(s1[5])

    elif node['ComponentType'] == 'NormalizeComponent':
        node['InputDim'] = int(s1[3])
        node['TargetRms'] = float(s1[5])
        node['AddLogStddev'] = (s1[7] == 'T')

    elif node['ComponentType'] == 'FixedScaleComponent':
        f = lines[0].index('[')
        node['Scales'] = str2num(lines[0][f+1:-2])
        shift = 2

    elif node['ComponentType'] == 'LogSoftmaxComponent':
        node['Dim'] = int(s1[3])
        shift = 3

    else:
        raise('Unsupported component type "' + node['ComponentType'] + '"')

    return node, shift


def find(s, ch1, ch2):
    return [i for i, ltr in enumerate(s) if ltr == ch1 or ltr == ch2]

def parseInputNode(line):
    #parse node line and return a dict

    node = {}
    f = find(line, ' ', '=')
    node['name'] = line[f[1]+1:f[2]]
    node['dim'] = int(line[f[3]+1:])
    return node


def parseComponentNode(line):
    #parse node line and return a dict
    node = {}
    f = find(line, ' ', '=')
    node['name'] = line[f[1]+1:f[2]]
    node['component'] = line[f[3]+1:f[4]]
    node['input'] = line[f[5]+1:]
    return node


def parseOutputNode(line):
    #parse node line and return a dict
    node = {}
    f = find(line, ' ', '=')
    node['name'] = line[f[1]+1:f[2]]
    node['input'] = line[f[5]+1:]
    node['objective'] = line[f[5]+1:]
    return node


#main function to read a NNET3 txt model
def readNNet3Txt(filename):
    lines = open(filename).readlines()

    input_nodes = []
    component_nodes = []
    output_nodes = []
    n = 1
    while lines[n].strip() and lines[n][0:16] != '<NumComponents> ':
        s1 = lines[n].strip().split()

        if s1[0] == 'input-node':
            input_nodes.append(parseInputNode(lines[n].rstrip()))
        elif s1[0] == 'component-node':
            component_nodes.append(parseComponentNode(lines[n].rstrip()))
        elif s1[0] == 'output-node':
            output_nodes.append(parseOutputNode(lines[n].rstrip()))
        else:
            warn('Unrecognized node "' + s1[0] + '"')
        n += 1

    if not lines[n].strip() and n - 1 < len(lines):
        n += 1

    #<NumComponents> 25
    if len(lines) < n - 1 or not lines[n].strip() or lines[n][0:16] != '<NumComponents> ':
        raise('Wrong format - <NumComponents> missing - must be the Kaldi Nnet3 txt format')

    NC = int(lines[n][16:])
    n = n + 1
    components = {}
    for k in range(NC):
        comp, shift = parseComponent(lines[n:])
        components[comp['ComponentName']] = comp
        n = n + shift

    nn = {}
    nn["input_nodes"] = input_nodes
    nn["component_nodes"] = component_nodes
    nn["output_nodes"] = output_nodes
    nn["components"] = components

    return nn