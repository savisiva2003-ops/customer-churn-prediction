import pymongo
from datetime import datetime
import os

def test_mongodb_insert():
    try:
        # Use environment variable for connection string
        # Set this with: export MONGODB_URI="your_connection_string"
        uri = os.environ.get(
            "MONGODB_URI", 
            "mongodb://localhost:27017"  # Default fallback for local testing
        )
        
        # Connect to MongoDB with 5 seconds timeout
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client["churn"]
        collection = db["customer_data"]

        # Create a test document
        test_doc = {
            "test_name": "HelloWorldTest",
            "timestamp": datetime.utcnow()
        }

        # Insert the document
        result = collection.insert_one(test_doc)

        print("✅ Test Insert Successful!")
        print("Inserted ID:", result.inserted_id)

    except Exception as e:
        print("❌ Test Insert Failed!")
        print(e)

# Run the test
if __name__ == "__main__":
    test_mongodb_insert()