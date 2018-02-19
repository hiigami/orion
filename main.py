from os.path import dirname, join
from dotenv import load_dotenv

from DataGetters.reddit_task import main

dotenv_path = join(dirname(__file__), 'config/.env')
load_dotenv(dotenv_path)

if __name__ == '__main__':
    main()