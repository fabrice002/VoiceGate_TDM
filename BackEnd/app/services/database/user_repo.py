# app/services/database/user_repo.py
"""
User database repository
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import logging

from models.user import UserCreate, UserInDB, UserUpdate, UserResponse
from core.database import get_db  # Use get_db instead of db directly

logger = logging.getLogger(__name__)


class UserRepository:
    """
    Repository for user database operations
    """
    
    def __init__(self):
        # Get database connection using get_db() to ensure it's initialized
        self.db = get_db()
        self.collection = self.db.users if hasattr(self.db, 'users') else None
        
        if self.collection is None:
            logger.warning("Users collection not available. Database may not be properly initialized.")
    
    def create(self, user_data: UserCreate) -> Tuple[bool, Optional[UserInDB], str]:
        """
        Create a new user
        
        Args:
            user_data: User creation data
            
        Returns:
            Tuple[bool, Optional[UserInDB], str]: (success, user_object, message)
        """
        try:
            # Check if collection is available
            if self.collection is None:
                logger.error("Database collection not available")
                return False, None, "Database not available"
            
            # Check if user exists
            existing = self.get_by_username(user_data.username)
            if existing:
                return False, None, f"User '{user_data.username}' already exists"
            
            # Create user document
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
                "metadata": {}
            }
            
            # Insert into database
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
        Get user by username
        
        Args:
            username: Username to search for
            
        Returns:
            Optional[UserInDB]: User object if found, None otherwise
        """
        try:
            if self.collection is None:
                return None
                
            user_doc = self.collection.find_one({"username": username.lower()})
            if user_doc:
                # Ensure _id is converted to string
                if "_id" in user_doc:
                    user_doc["_id"] = str(user_doc["_id"])
                return UserInDB(**user_doc)
            return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        """
        Get user by ID
        
        Args:
            user_id: User ID to search for
            
        Returns:
            Optional[UserInDB]: User object if found, None otherwise
        """
        try:
            if self.collection is None:
                return None
                
            user_doc = self.collection.find_one({"_id": user_id})
            if user_doc:
                # Ensure _id is converted to string
                if "_id" in user_doc:
                    user_doc["_id"] = str(user_doc["_id"])
                return UserInDB(**user_doc)
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def update_voice_embedding(self, username: str, embedding: List[float]) -> bool:
        """
        Update user's voice embedding
        
        Args:
            username: Username
            embedding: Voice embedding vector
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if self.collection is None:
                return False
                
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
        Get all users with pagination
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List[UserInDB]: List of users
        """
        try:
            if self.collection is None:
                return []
                
            cursor = self.collection.find().skip(skip).limit(limit)
            users = []
            for doc in cursor:
                # Ensure _id is converted to string
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                users.append(UserInDB(**doc))
            return users
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []
    
    def update(self, username: str, update_data: UserUpdate) -> Tuple[bool, Optional[UserInDB], str]:
        """
        Update user information
        
        Args:
            username: Username
            update_data: Update data
            
        Returns:
            Tuple[bool, Optional[UserInDB], str]: (success, user_object, message)
        """
        try:
            if self.collection is None:
                return False, None, "Database not available"
                
            user = self.get_by_username(username)
            if not user:
                return False, None, f"User '{username}' not found"
            
            # Prepare update
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
            
            # Update database
            result = self.collection.update_one(
                {"_id": user.id},
                {"$set": update_dict}
            )
            
            if result.modified_count > 0:
                # Get updated user
                updated_user = self.get_by_id(user.id)
                logger.info(f"Updated user: {username}")
                return True, updated_user, "User updated successfully"
            
            return False, user, "No changes made"
            
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False, None, f"Failed to update user: {str(e)}"
    
    def delete(self, username: str) -> bool:
        """
        Delete a user
        
        Args:
            username: Username to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            if self.collection is None:
                return False
                
            result = self.collection.delete_one({"username": username.lower()})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False
    
    def get_voice_registered_users(self) -> List[str]:
        """
        Get list of users with registered voices
        
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