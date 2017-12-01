

import numpy as np
from warnings import warn
import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")



def CheckToken(token):
    for ch in token:
        if ch <= ' ':
            raise Exception('Token cannot contain whitespaces')


def KaldiReaderString(fid):
    str = u''
    ch = fid.read(1).decode('utf-8')
    while ch != u' ':
        str = str + ch
        ch = fid.read(1).decode('utf-8')
    return str


def KaldiReaderExpectToken(fid, binary, token):
    if len(token) == 0:
        Exception('Empty token')
    CheckToken(token)
    str = KaldiReaderString(fid)

    if len(str) <= 1:
        raise Exception('Failed to read token ' + token)
    if str != token:
        raise Exception('Expected token ""' + token + '"", got instead ""' + str + '""')


def KaldiReaderReadBasicType(fid, binary, dtype):
    if not binary:
        raise Exception('Txt format not implemented')
    c = int.from_bytes(fid.read(1), byteorder='little')
    s = np.dtype(dtype).itemsize
    if c != s:
        raise Exception('Data type ""' + np.dtype(dtype).name + '"" do not match with item size %d' % c)
    x = np.fromfile(fid, dtype, 1)
    return x[0]


def KaldiReaderReadVector(fid, binary):
    if not binary:
        raise Exception('Txt format not implemented')
    type = KaldiReaderString(fid)
    if type[1] != 'V':
        raise Exception('Data item is not vector')
    dim = KaldiReaderReadBasicType(fid, binary, np.int32)
    if type[0] == 'F':
        x = np.fromfile(fid, np.float32, dim)
    else:
        x = np.fromfile(fid, np.float64, dim)
    if x.size != dim:
        raise Exception('Error during reading of vector data')
    return x


def KaldiReaderReadMatrix(fid, binary):
    if not binary:
        raise Exception('Txt format not implemented')
    type = KaldiReaderString(fid)
    if type[0] == 'C':
        raise Exception('Compressed format not implemented')
    if type[1] != 'M':
        raise Exception('Data item is not matrix')
    rows = KaldiReaderReadBasicType(fid, binary, np.int32)
    cols = KaldiReaderReadBasicType(fid, binary, np.int32)
    if type[0] == 'F':
        x = np.fromfile(fid, np.float32, rows*cols)
    else:
        x = np.fromfile(fid, np.float64, rows*cols)
    if x.size != rows*cols:
        raise Exception('Error during reading of matrix data')
    x = np.reshape(x, (rows, cols))
    return x


def KaldiReaderReadSpMatrix(fid, binary):
    if not binary:
        raise Exception('Txt format not implemented')
    type = KaldiReaderString(fid)
    if type[0] == '[':
        raise Exception('New format not supported')
    if type[1] != 'P':
        raise Exception('Data item is not symetric-packed matrix')
    rows = KaldiReaderReadBasicType(fid, binary, np.int32)
    num = int(((rows+1)*rows)/2)
    if type[0] == 'F':
        x = np.fromfile(fid, np.float32, num)
    else:
        x = np.fromfile(fid, np.float64, num)
    if x.size != num:
        raise Exception('Error during reading of matrix data')
    return x


def str2num(s):
    floats = [float(i) for i in s.split()]
    floats = np.asarray(floats, dtype=np.float32)
    return floats


def parseComponent(lines):
    #parse component and return a dict and number of "consumed" lines
    if lines[0][0:15] != '<ComponentName>':
        raise Exception('Wrong format - <ComponentName> missing - must be the Kaldi Nnet3 txt format')

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
        for k in range(1, n-1):
            node['LinearParams'][k-1, :] = str2num(lines[k])
        node['LinearParams'][-1, :] = str2num(lines[n-1][0:-2])
        if lines[n][0:12] != '<BiasParams>':
            raise Exception('Wrong format - <BiasParams> missing in <FixedAffineComponent> - must be the Kaldi Nnet3 txt format')
        node['BiasParams'] = str2num(lines[n][15:-2])
        shift = n + 2

    elif node['ComponentType'] == 'RectifiedLinearComponent':
        node['InputDim'] = int(s1[3])
        shift = 3

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
        raise Exception('Unsupported component type "' + node['ComponentType'] + '"')

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
    node['input'] = line[f[3]+1:f[4]]
    node['objective'] = line[f[5]+1:]
    return node


def readNNet3Txt(filename):
    # main function to read a NNET3 txt model
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
        raise Exception('Wrong format - <NumComponents> missing - must be the Kaldi Nnet3 txt format')

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


def calculateGConsts(w, miv, iv):
    dim = miv.shape[1]
    offset = -0.5 * np.log(2*np.pi) * dim
    N = w.numel
    gc = np.ones([N], dtype=np.float32) * offset
    for k, g in enumerate(w):
        gc[k] = gc[k] + np.sum(0.5 * np.log(iv[k]) - 0.5 * miv[k] * miv[k] / iv[k])
    return gc


def readGMM(filename):
    # main function to read a GMM model - return gc, w, miv, iv
    fid = open(filename, 'rb')
    ch = fid.read(2).decode('utf-8')
    if ch == '\x00' + 'B':
        binary = True
    else:
        fid.close()
        raise Exception('GMM is not in binary format')

    KaldiReaderExpectToken(fid, binary, '<DiagGMM>')
    try:
        KaldiReaderExpectToken(fid, binary, '<GCONSTS>')  # optional
        gc = KaldiReaderReadVector(fid, binary)
        calculateGC = False
    except ValueError:
        calculateGC = True

    KaldiReaderExpectToken(fid, binary, '<WEIGHTS>')
    w = KaldiReaderReadVector(fid, binary)
    KaldiReaderExpectToken(fid, binary, '<MEANS_INVVARS>')
    miv = KaldiReaderReadMatrix(fid, binary)
    KaldiReaderExpectToken(fid, binary, '<INV_VARS>')
    iv = KaldiReaderReadMatrix(fid, binary)
    KaldiReaderExpectToken(fid, binary, '</DiagGMM>')
    fid.close()

    if calculateGC:
       gc = calculateGConsts(w, miv, iv)

    return [gc, w, miv, iv]


def readIvectorsExtractor(filename):
    # main function to read a ivectorsExtractor data - return data_dim, ivec_dim, w, w_vec, M, sigma_inv, prior_offset
    fid = open(filename, 'rb')
    ch = fid.read(2).decode('utf-8')
    if ch == '\x00' + 'B':
        binary = True
    else:
        fid.close()
        raise Exception('ivectorsExtractor is not in binary format')

    KaldiReaderExpectToken(fid, binary, '<IvectorExtractor>')
    KaldiReaderExpectToken(fid, binary, '<w>')
    w = KaldiReaderReadMatrix(fid, binary)
    KaldiReaderExpectToken(fid, binary, '<w_vec>')
    w_vec = KaldiReaderReadVector(fid, binary)
    KaldiReaderExpectToken(fid, binary, '<M>')
    numGauss = KaldiReaderReadBasicType(fid, binary, np.int32)
    if numGauss <= 0:
        raise Exception('Wrong number of Gaussians in ivector extractor data')
    M = [KaldiReaderReadMatrix(fid, binary)]
    for i in range(1, numGauss):
        M.append(KaldiReaderReadMatrix(fid, binary))
    KaldiReaderExpectToken(fid, binary, '<SigmaInv>')
    sigma_inv = []
    for i in range(numGauss):
        sigma_inv.append(KaldiReaderReadSpMatrix(fid, binary))
    KaldiReaderExpectToken(fid, binary, '<IvectorOffset>')
    prior_offset = KaldiReaderReadBasicType(fid, binary, np.float64)
    KaldiReaderExpectToken(fid, binary, '</IvectorExtractor>')

    fid.close()

    data_dim = M[0].shape[0]
    ivec_dim = M[0].shape[1]

    return [data_dim, ivec_dim, w, w_vec, M, sigma_inv, prior_offset]


def iePostprocess(ie):
    # calculate sigma_inv * M and U matrices
    data_dim = ie[0]
    ivec_dim = ie[1]
    M = ie[4]
    sigma_inv = ie[5]
    sigma_inv_M = []
    U = []
    for i in range(len(ie[4])):
        s_full = np.zeros([data_dim, data_dim], dtype=sigma_inv[0].dtype)
        s_full[np.tril_indices(data_dim)] = sigma_inv[i]
        s_full = s_full + np.transpose(s_full) - np.diag(np.diag(s_full))
        sigma_inv_M.append(np.matmul(s_full, M[i]))
        Utmp = np.matmul(np.matmul(np.transpose(M[i]), s_full), M[i])
        U.append(Utmp[np.tril_indices(ivec_dim)])

    return sigma_inv_M, U