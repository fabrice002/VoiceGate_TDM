from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MockCollection:
    """Mock MongoDB collection with file persistence"""
    
    def __init__(self, name: str):
        self.name = name
        self._data_file = Path(f"data/mock_db/{name}.json")
        self._data_file.parent.mkdir(parents=True, exist_ok=True)
        self._documents: List[Dict] = []
        self._counter = 1
        self._indexes = {}
        self._load_data()
    
    def _load_data(self):
        """Load data from file"""
        try:
            if self._data_file.exists():
                with open(self._data_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._documents = data
                        # Find max ID
                        max_id = 0
                        for doc in self._documents:
                            if "_id" in doc:
                                try:
                                    doc_id = int(doc["_id"])
                                    if doc_id > max_id:
                                        max_id = doc_id
                                except:
                                    pass
                        self._counter = max_id + 1
                    else:
                        self._documents = []
                logger.info(f"Loaded {len(self._documents)} documents from {self.name}")
        except Exception as e:
            logger.error(f"Error loading data from {self._data_file}: {e}")
            self._documents = []
    
    def _save_data(self):
        """Save data to file"""
        try:
            with open(self._data_file, 'w') as f:
                json.dump(self._documents, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Error saving data to {self._data_file}: {e}")
    
    def insert_one(self, document: Dict) -> Any:
        """Insert document"""
        doc = document.copy()
        if "_id" not in doc:
            doc["_id"] = str(self._counter)
            self._counter += 1
        else:
            doc["_id"] = str(doc["_id"])
        
        # Check unique indexes
        for field, unique in self._indexes.items():
            if unique and field in doc:
                existing = self.find_one({field: doc[field]})
                if existing and existing["_id"] != doc["_id"]:
                    raise Exception(f"Duplicate key error on field: {field}")
        
        self._documents.append(doc)
        self._save_data()
        
        # Create mock InsertResult
        class InsertResult:
            def __init__(self, inserted_id):
                self.inserted_id = inserted_id
        
        return InsertResult(doc["_id"])
    
    def find_one(self, query: Optional[Dict] = None) -> Optional[Dict]:
        """Find one document"""
        if not query:
            return self._documents[0].copy() if self._documents else None
        
        for doc in self._documents:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc.copy()
        return None
    
    def find(self, query: Optional[Dict] = None, projection: Optional[Dict] = None):
        """Find documents"""
        results = []
        if not query:
            results = [d.copy() for d in self._documents]
        else:
            results = [d.copy() for d in self._documents 
                      if all(d.get(k) == v for k, v in query.items())]
        
        if projection:
            results = [{k: d[k] for k in projection if k in d} for d in results]
        
        return MockCursor(results)
    
    def update_one(self, query: Dict, update: Dict) -> Any:
        """Update document"""
        modified_count = 0
        for doc in self._documents:
            if all(doc.get(k) == v for k, v in query.items()):
                if "$set" in update:
                    doc.update(update["$set"])
                if "$inc" in update:
                    for key, value in update["$inc"].items():
                        doc[key] = doc.get(key, 0) + value
                modified_count = 1
                break
        
        if modified_count:
            self._save_data()
        
        # Create mock UpdateResult
        class UpdateResult:
            def __init__(self, modified_count):
                self.modified_count = modified_count
        
        return UpdateResult(modified_count)
    
    def delete_one(self, query: Dict) -> Any:
        """Delete document"""
        deleted_count = 0
        for i, doc in enumerate(self._documents):
            if all(doc.get(k) == v for k, v in query.items()):
                self._documents.pop(i)
                deleted_count = 1
                self._save_data()
                break
        
        # Create mock DeleteResult
        class DeleteResult:
            def __init__(self, deleted_count):
                self.deleted_count = deleted_count
        
        return DeleteResult(deleted_count)
    
    def count_documents(self, query: Optional[Dict] = None) -> int:
        """Count documents"""
        if not query:
            return len(self._documents)
        return sum(1 for d in self._documents 
                  if all(d.get(k) == v for k, v in query.items()))


class MockCursor:
    """Mock cursor for iterating results"""
    
    def __init__(self, documents: List[Dict]):
        self._documents = documents
    
    def sort(self, *args, **kwargs):
        return self
    
    def limit(self, n: int):
        self._documents = self._documents[:n]
        return self
    
    def skip(self, n: int):
        self._documents = self._documents[n:]
        return self
    
    def __iter__(self):
        return iter(self._documents)


class MockDatabase:
    """Mock database with proper initialization"""
    
    def __init__(self):
        self._collections = {}
        self._connected = False  # Add _connected attribute for mock
        self._init_collections()
    
    def _init_collections(self):
        """Initialize default collections"""
        self._collections["users"] = MockCollection("users")
        self._collections["conversations"] = MockCollection("conversations")
        self._collections["voice_samples"] = MockCollection("voice_samples")
    
    def __getattr__(self, name: str):
        """Get collection dynamically"""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]
    
    def __getitem__(self, name: str):
        """Get collection by name"""
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]


class DatabaseConnection:
    """Database connection manager"""
    
    def __init__(self):
        self._client = None
        self._db = None
        self._connected = False
        self._initialized = False
    
    def connect(self):
        """Connect to database"""
        from core.config import settings
        
        if not self._initialized:
            # Ensure data directories exist
            os.makedirs("data/mock_db", exist_ok=True)
            os.makedirs(settings.VOICE_DB_FOLDER, exist_ok=True)
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            self._initialized = True
        
        if settings.USE_MONGODB and settings.MONGODB_URI:
            try:
                from pymongo import MongoClient, ASCENDING
                from pymongo.server_api import ServerApi
                
                logger.info(f"Attempting to connect to MongoDB at {settings.MONGODB_URI}...")
                
                self._client = MongoClient(
                    settings.MONGODB_URI,
                    server_api=ServerApi("1"),
                    serverSelectionTimeoutMS=5000
                )
                
                # Test connection
                self._client.admin.command("ping")
                self._db = self._client[settings.MONGO_DB_NAME]
                self._connected = True
                
                logger.info(f"✓ Connected to MongoDB: {settings.MONGO_DB_NAME}")
                self._create_indexes()
                
                return self._db
                
            except Exception as e:
                logger.warning(f"✗ MongoDB connection failed: {e}")
                logger.info("Falling back to mock database with file persistence")
                self._db = MockDatabase()
                self._connected = False
        else:
            logger.info("Using mock database with file persistence (MongoDB disabled)")
            self._db = MockDatabase()
            self._connected = False
        
        # Ensure default collections exist
        self._ensure_collections()
        
        return self._db
    
    def _ensure_collections(self):
        """Ensure all required collections exist"""
        try:
            # These will create collections on first access
            _ = self._db.users
            _ = self._db.voice_samples
            _ = self._db.conversations
            logger.info("✓ All collections initialized")
        except Exception as e:
            logger.error(f"Error ensuring collections: {e}")
    
    def _create_indexes(self):
        """Create database indexes"""
        if not self._connected or not self._db:
            return
        
        try:
            from pymongo import ASCENDING
            
            # Users indexes
            self._db.users.create_index([("username", ASCENDING)], unique=True)
            self._db.users.create_index([("created_at", ASCENDING)])
            
            # Conversations indexes
            self._db.conversations.create_index([
                ("user_id", ASCENDING),
                ("timestamp", -1)
            ])
            
            logger.info("Database indexes created")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def close(self):
        """Close connection"""
        if self._client:
            self._client.close()
            logger.info("Database connection closed")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MongoDB"""
        return self._connected


# Global instance
db_connection = DatabaseConnection()
db = None  # Will be initialized in main.py