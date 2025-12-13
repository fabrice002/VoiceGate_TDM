# app/api/routes/users.py

from fastapi import APIRouter, HTTPException, Query
import logging

from models.user import UserCreate
from schemas.responses import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=APIResponse)
def create_user(user_data: UserCreate):
    """Create new user"""
    try:
        from services.database.user_repo import UserRepository
        user_repo = UserRepository()
        
        success, user, message = user_repo.create(user_data)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return APIResponse(
            success=True,
            message=message,
            data={"user_id": user.id, "username": user.username}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all users"""
    try:
        from services.database.user_repo import UserRepository
        user_repo = UserRepository()
        
        skip = (page - 1) * limit
        users = user_repo.get_all(skip, limit)
        
        return {
            "total": len(users),
            "page": page,
            "limit": limit,
            "users": [u.model_dump() for u in users]
        }
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{username}")
def get_user(username: str):
    """Get user by username"""
    try:
        from services.database.user_repo import UserRepository
        user_repo = UserRepository()
        
        user = user_repo.get_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        
        return user.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    """
    Supprime un utilisateur et ses données associées.
    """
    try:
        user_repo = UserRepository()
        
        # 1. Vérifier si l'utilisateur existe
        user = user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        # 2. Supprimer l'utilisateur
        # Assurez-vous que votre UserRepository a une méthode delete !
        # Si elle n'existe pas, voici à quoi elle ressemblerait :
        # db.users.delete_one({"_id": user_id})
        success = user_repo.delete_user(user_id)
        
        if not success:
             raise HTTPException(status_code=500, detail="Failed to delete user")
             
        # Optionnel : Nettoyer les conversations orphelines ici
        conversation_repo.delete_all_for_user(user_id)

        return None # 204 No Content

    except HTTPException:
        raise
    except Exception as e:
        # Logger l'erreur réelle
        print(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))