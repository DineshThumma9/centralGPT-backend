"""
Unit tests for schema models and validation.
"""
import pytest
from datetime import datetime
from uuid import uuid4
from pydantic import ValidationError

from src.models.schema import (
    ChatMessage, 
    SenderRole, 
    UserPayload,
    MsgRequest,
    GitRequest
)


class TestChatMessage:
    """Test ChatMessage model validation."""
    
    def test_valid_chat_message(self):
        """Test creating a valid chat message."""
        message = ChatMessage(
            role=SenderRole.USER,
            content="Hello, world!"
        )
        assert message.role == SenderRole.USER
        assert message.content == "Hello, world!"
        assert message.timestamp is None  # Optional field
    
    def test_chat_message_with_timestamp(self):
        """Test chat message with timestamp."""
        now = datetime.now()
        message = ChatMessage(
            role=SenderRole.ASSISTANT,
            content="Response",
            timestamp=now
        )
        assert message.timestamp == now
    
    def test_invalid_role(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid_role", content="test")


class TestUserPayload:
    """Test UserPayload model validation."""
    
    def test_valid_user_payload(self):
        """Test creating a valid user payload."""
        user = UserPayload(
            username="testuser",
            email="test@example.com",
            password="securepassword123"
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password == "securepassword123"
    
    def test_invalid_email(self):
        """Test that invalid email raises validation error."""
        with pytest.raises(ValidationError):
            UserPayload(
                username="testuser",
                email="not-an-email",
                password="password"
            )


class TestMsgRequest:
    """Test MsgRequest model validation."""
    
    def test_valid_msg_request(self):
        """Test creating a valid message request."""
        session_id = str(uuid4())
        msg_request = MsgRequest(
            session_id=session_id,
            msg="Hello",
            context_id="ctx123",
            context_type="general"
        )
        assert msg_request.session_id == session_id
        assert msg_request.isFirst is False  # Default value
        assert msg_request.files is None  # Optional field
    
    def test_msg_request_with_files(self):
        """Test message request with files."""
        msg_request = MsgRequest(
            session_id=str(uuid4()),
            msg="Check these files",
            context_id="ctx123",
            context_type="file_analysis",
            files=["file1.pdf", "file2.txt"]
        )
        assert len(msg_request.files) == 2


class TestGitRequest:
    """Test GitRequest model validation."""
    
    def test_minimal_git_request(self):
        """Test creating a minimal git request."""
        git_req = GitRequest(
            owner="testowner",
            repo="testrepo"
        )
        assert git_req.owner == "testowner"
        assert git_req.repo == "testrepo"
        assert git_req.branch == "main"  # Default value
        assert git_req.commit is None
    
    def test_git_request_with_filters(self):
        """Test git request with include/exclude filters."""
        git_req = GitRequest(
            owner="testowner",
            repo="testrepo",
            branch="develop",
            dir_include=["src", "tests"],
            file_extension_include=[".py", ".js"],
            file_extension_exclude=[".pyc", ".log"]
        )
        assert git_req.branch == "develop"
        assert "src" in git_req.dir_include
        assert ".py" in git_req.file_extension_include
        assert ".pyc" in git_req.file_extension_exclude