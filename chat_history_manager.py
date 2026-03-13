"""
Chat History Manager for MongoDB Integration
"""
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from flask import session
from pymongo import ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self, db, retention_days=7):
        self.db = db
        self.retention_days = retention_days
        self._setup_collection()
    
    def _setup_collection(self):
        """Initialize chat history collection and indexes"""
        try:
            if not 'chat_history' in self.db.list_collection_names():
                self.db.create_collection('chat_history')
            
            chat_history = self.db.chat_history
            
            # Create indexes
            chat_history.create_index([("user_id", ASCENDING)])
            chat_history.create_index([("timestamp", DESCENDING)])
            
            # TTL index for auto-cleanup
            chat_history.create_index(
                [("expires_at", ASCENDING)],
                expireAfterSeconds=0,
                name="chat_history_ttl"
            )
            
            logger.info("Chat history collection initialized with indexes")
        except Exception as e:
            logger.error(f"Error setting up chat history collection: {e}")
    
    def get_chat_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get chat history from MongoDB"""
        try:
            # Get last 50 messages ordered by timestamp
            history = list(self.db.chat_history.find(
                {'user_id': user_id},
                {'_id': 0}  # Exclude MongoDB _id
            ).sort('timestamp', DESCENDING).limit(50))
            
            # Convert datetime objects to ISO format
            for msg in history:
                for key in ['timestamp', 'created_at', 'expires_at']:
                    if key in msg and isinstance(msg[key], datetime):
                        msg[key] = msg[key].isoformat()
            
            # Reverse to get chronological order
            return list(reversed(history))
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def add_to_chat_history(self, user_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to chat history in MongoDB"""
        try:
            current_time = datetime.utcnow()
            
            message = {
                'user_id': user_id,
                'role': role,
                'content': content,
                'timestamp': current_time,
                'created_at': current_time,
                'metadata': metadata or {},
                'expires_at': current_time + timedelta(days=self.retention_days)
            }
            
            result = self.db.chat_history.insert_one(message)
            logger.debug(f"Added message to MongoDB chat history for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to chat history: {e}")
            return False
    
    def clear_user_history(self, user_id: str):
        """Clear all chat history for a specific user"""
        try:
            result = self.db.chat_history.delete_many({'user_id': user_id})
            logger.info(f"Cleared {result.deleted_count} messages for user {user_id}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")
            return 0
    
    def export_user_history(self, user_id: str) -> Dict[str, Any]:
        """Export all chat history for a user"""
        try:
            history = list(self.db.chat_history.find(
                {'user_id': user_id},
                {'_id': 0}
            ).sort('timestamp', ASCENDING))
            
            # Convert datetime objects to ISO format
            for msg in history:
                for key in ['timestamp', 'created_at', 'expires_at']:
                    if key in msg and isinstance(msg[key], datetime):
                        msg[key] = msg[key].isoformat()
            
            return {
                'user_id': user_id,
                'exported_at': datetime.utcnow().isoformat(),
                'message_count': len(history),
                'history': history
            }
        except Exception as e:
            logger.error(f"Error exporting chat history: {e}")
            return {}