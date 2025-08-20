from dotenv import load_dotenv
import os
from urllib.parse import quote_plus

load_dotenv()

raw_mongodb_username =os.getenv("MONGO_DB_USERNAME")
raw_mongodb_password = os.getenv("MONGO_DB_PASSWORD")

mongodb_username =quote_plus(raw_mongodb_username)
mongodb_password = quote_plus(raw_mongodb_password)

from pymongo.mongo_client import MongoClient

uri = f"mongodb+srv://{mongodb_username}:{mongodb_password}@cluster0.yfmjcsk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)