import logging

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

def getLogger(name: str):
    logger = logging.getLogger(name)
    # ToDo (jordycuan) - Could this level be changed?
    logger.setLevel(logging.DEBUG) 
    logger.addHandler(console)
    return logger
