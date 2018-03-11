from pymongo import MongoClient
from getpass import getpass
import requests

MONGO_USER = None
MONGO_PWD = None
MONGO_DOM = "54.227.50.102"
MONGO_PORT = None
DB_NAME = "orion"

if (requests.get("http://jsonip.com").json()['ip'] == "54.227.50.102"):
	MONGO_DOM = "localhost"

def get_orion_db():
	global MONGO_USER, MONGO_PWD, MONGO_PORT
	print(MONGO_USER == MONGO_PWD == MONGO_PORT == None)
	if (MONGO_USER == MONGO_PWD == MONGO_PORT == None):
		MONGO_USER = input(">>> MongoDB username: ")
		MONGO_PWD = getpass(">>> MongoDB password: ")
		MONGO_PORT = input(">>> MongoDB Port: ")

	return MongoClient(
		"mongodb://" + MONGO_USER + ":" + MONGO_PWD +
		"@" + MONGO_DOM + ":" + MONGO_PORT + "/" + DB_NAME
	)[DB_NAME]
