import os
import numpy as np
from json import JSONEncoder
import json

class NumpyArrayEncoder(JSONEncoder):
    """
    Reference: https://pynative.com/python-serialize-numpy-ndarray-into-json/
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def foldercheck(Savepath):
    if(not (os.path.isdir(Savepath))):
        print(Savepath, "  was not present, creating the folder...")
        os.makedirs(Savepath)


def saveSIFTmatches(i, pts1_, pts2_, siftFilepath = './data/sift/'):
    foldercheck(siftFilepath)
    siftpoints = {}
    siftpoints['pts1'] = pts1_
    siftpoints['pts2'] = pts2_
    
    siftFileName = siftFilepath + 'frame_'+str(i)+'.json'
    with open(siftFileName, 'w') as f:
        json.dump(siftpoints, f, cls=NumpyArrayEncoder)

def loadSIFTmatches(i, siftFilepath = './data/sift/'):
    
    siftFileName = siftFilepath + 'frame_'+str(i)+'.json'
    with open(siftFileName, 'r') as f:
        siftpoints = json.load(f)
    pts1_ = siftpoints['pts1']
    pts2_ = siftpoints['pts2']
    
    return np.array(pts1_), np.array(pts2_)

        
def saveDict(record, name, Filepath = './data/record/'):
    foldercheck(Filepath)    
    FileName = Filepath + name + '.json'
    with open(FileName, 'w') as f:
        json.dump(record, f, cls=NumpyArrayEncoder)


def loadDict(name, Filepath = './data/record/'):

    """
    if there are poses, return the data
    else, write and return empty dictionary 
    """
    foldercheck(Filepath)
    FileName = Filepath + name + '.json'
    
    if os.path.exists(FileName):
        with open(FileName, 'r') as f:
            files = json.load(f)
    else:
        print(FileName, 'not found, creating json')
        f = open(FileName, "w")
        f.write('{}')
        files = {}
        
    return files

