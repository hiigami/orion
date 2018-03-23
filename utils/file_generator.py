import json

def readfile_generator(filepath, process_line=None, limit=0):
    with open(filepath) as fp:
        count = 0
        while True:
            line = fp.readline()
            if not line:
                break
            if limit and count >= limit:
                break
            
            count += 1 
            if (process_line):
                yield process_line(line)
            else:
                yield line

def process_line(line):     
    return json.loads(line.strip()[:-1])
