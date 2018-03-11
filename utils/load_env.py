from os.path import dirname, join

from dotenv import load_dotenv


def load():
    dotenv_path = join(dirname(__file__), '../config/.env')
    load_dotenv(dotenv_path)
