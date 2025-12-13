# app/services/database/user_repo.py
"""
User database repository with Offline/Mock support
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import logging
import uuid

from models.user import UserCreate, UserInDB, UserUpdate, UserResponse

logger = logging.getLogger(__name__)

class UserRepository:
    """
    Repository for user database operations.
    Handles both MongoDB connections and Mock/Offline fallbacks.
    """
    
    def __init__(self):
        """
        Initialize the repository.
        Attempts to connect to MongoDB via core.database.
        Falls back to Mock Mode if database is unavailable.
        """
        try:
            # Dynamic import to avoid circular dependencies
            from core.database import get_db, db_connection
            
            # 1. Try get_db() first (FastAPI dependency style)
            self.db = get_db()
            
            # 2. Fallback to direct connection object if get_db returns None
            if self.db is None:
                self.db = getattr(db_connection, 'db', None) or getattr(db_connection, '_db', None)

            # 3. Initialize collection
            if self.db is not None:
                self.collection = self.db.users
            else:
                self.collection = None
                logger.warning("⚠️ Users collection not available. Running in Mock Mode.")
                
        except Exception as e:
            logger.error(f"Error initializing UserRepository: {e}")
            self.collection = None
    
    def create(self, user_data: UserCreate) -> Tuple[bool, Optional[UserInDB], str]:
        """
        Create a new user in the database.
        
        Args:
            user_data: Object containing user creation details (username, email)
            
        Returns:
            Tuple[bool, Optional[UserInDB], str]: 
                - Success flag
                - Created user object (or None)
                - Status message
        """
        try:
            # --- MOCK MODE ---
            if self.collection is None:
                logger.info(f"[MOCK] Creating user: {user_data.username}")
                mock_user = UserInDB(
                    id=str(uuid.uuid4()),
                    username=user_data.username.lower(),
                    email=user_data.email,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    role="user",
                    status="active",
                    is_voice_registered=False,
                    voice_samples_count=0
                )
                return True, mock_user, "User created successfully (Mock)"

            # --- REAL DB MODE ---
            # Check if user exists
            existing = self.get_by_username(user_data.username)
            if existing:
                return False, None, f"User '{user_data.username}' already exists"
            
            user_doc = {
                "username": user_data.username.lower(),
                "email": user_data.email,
                "voice_embedding": None,
                "voice_samples_count": 0,
                "is_voice_registered": False,
                "role": "user",
                "status": "active",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "last_voice_activity": None,
                "last_interaction_at": None, # For Wake Word session tracking
                "metadata": {}
            }
            
            result = self.collection.insert_one(user_doc)
            user_doc["_id"] = str(result.inserted_id)
            
            user = UserInDB(**user_doc)
            logger.info(f"Created user: {user.username}")
            
            return True, user, "User created successfully"
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, None, f"Failed to create user: {str(e)}"
    
    def get_by_username(self, username: str) -> Optional[UserInDB]:
        """
        Retrieve a user by their username.
        
        Args:
            username: The username to search for
            
        Returns:
            Optional[UserInDB]: The user object if found, else None
        """
        try:
            if self.collection is None:
                return None
                
            user_doc = self.collection.find_one({"username": username.lower()})
            if user_doc:
                if "_id" in user_doc:
                    user_doc["_id"] = str(user_doc["_id"])
                return UserInDB(**user_doc)
            return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        """
        Retrieve a user by their database ID.
        
        Args:
            user_id: The unique ID string
            
        Returns:
            Optional[UserInDB]: The user object if found, else None
        """
        try:
            if self.collection is None:
                return None
                
            user_doc = self.collection.find_one({"_id": user_id})
            if user_doc:
                if "_id" in user_doc:
                    user_doc["_id"] = str(user_doc["_id"])
                return UserInDB(**user_doc)
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def update_voice_embedding(self, username: str, embedding: List[float]) -> bool:
        """
        Update the voice biometric embedding for a user.
        
        Args:
            username: The user to update
            embedding: List of floats representing the voice vector
            
        Returns:
            bool: True if the update was successful
        """
        try:
            if self.collection is None:
                logger.info(f"[MOCK] Updated voice embedding for {username}")
                return True
                
            result = self.collection.update_one(
                {"username": username.lower()},
                {
                    "$set": {
                        "voice_embedding": embedding,
                        "is_voice_registered": True,
                        "last_voice_activity": datetime.now(),
                        "updated_at": datetime.now()
                    },
                    "$inc": {"voice_samples_count": 1}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating voice embedding: {e}")
            return False
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        """
        Get all users with pagination.
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List[UserInDB]: List of user objects
        """
        try:
            if self.collection is None:
                return []
                
            cursor = self.collection.find().skip(skip).limit(limit)
            users = []
            for doc in cursor:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                users.append(UserInDB(**doc))
            return users
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []
    
    def update(self, username: str, update_data: UserUpdate) -> Tuple[bool, Optional[UserInDB], str]:
        """
        Update generic user information (email, status, etc.).
        
        Args:
            username: Target username
            update_data: Object containing fields to update
            
        Returns:
            Tuple[bool, Optional[UserInDB], str]: Success flag, updated user, message
        """
        try:
            if self.collection is None:
                return False, None, "Database not available"
                
            user = self.get_by_username(username)
            if not user:
                return False, None, f"User '{username}' not found"
            
            # Prepare update dictionary
            update_dict = {}
            if update_data.email is not None:
                update_dict["email"] = update_data.email
            if update_data.status is not None:
                update_dict["status"] = update_data.status.value
            if update_data.metadata is not None:
                update_dict["metadata"] = update_data.metadata
            
            if not update_dict:
                return False, user, "No changes provided"
            
            update_dict["updated_at"] = datetime.now()
            
            # Perform Update
            result = self.collection.update_one(
                {"_id": user.id},
                {"$set": update_dict}
            )
            
            if result.modified_count > 0:
                updated_user = self.get_by_id(user.id)
                logger.info(f"Updated user: {username}")
                return True, updated_user, "User updated successfully"
            
            return False, user, "No changes made"
            
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False, None, f"Failed to update user: {str(e)}"

    def delete_user(self, user_id: str) -> bool:
        """
        Permanently delete a user from the database.
        
        Args:
            user_id: The ID of the user to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            if self.collection is None:
                logger.info(f"[MOCK] Deleted user {user_id}")
                return True
                
            result = self.collection.delete_one({"_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False

    def get_voice_registered_users(self) -> List[str]:
        """
        Retrieve a list of usernames that have voice prints registered.
        Used for 1:N Identification.
        
        Returns:
            List[str]: List of usernames
        """
        try:
            if self.collection is None:
                return []
                
            users = self.collection.find(
                {"is_voice_registered": True},
                {"username": 1}
            )
            return [user["username"] for user in users]
        except Exception as e:
            logger.error(f"Error getting registered users: {e}")
            return []

    # =========================================================
    #  WAKE WORD SESSION MANAGEMENT
    # =========================================================

    def update_last_interaction(self, user_id: str) -> bool:
        """
        Update the 'last_interaction_at' timestamp for the user.
        Used to reset the active session timer (e.g., 4 min window).
        
        Args:
            user_id: The user ID
            
        Returns:
            bool: True if update successful
        """
        try:
            if self.collection is None:
                return True
            
            self.collection.update_one(
                {"_id": user_id},
                {"$set": {"last_interaction_at": datetime.now()}}
            )
            return True
        except Exception as e:
            logger.error(f"Error updating interaction time: {e}")
            return False

    def get_last_interaction(self, user_id: str) -> Optional[datetime]:
        """
        Retrieve the 'last_interaction_at' timestamp.
        Used to check if the session is still valid.
        
        Args:
            user_id: The user ID
            
        Returns:
            Optional[datetime]: The timestamp of last activity, or None
        """
        try:
            if self.collection is None:
                return None
                
            user = self.collection.find_one({"_id": user_id}, {"last_interaction_at": 1})
            return user.get("last_interaction_at") if user else None
        except Exception as e:
            logger.error(f"Error getting interaction time: {e}")
            return None