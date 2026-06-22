import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from feedback.event_handlers import shift_vector, FeedbackHandler
from feedback.producer import FeedbackProducer
from feedback.consumer import FeedbackConsumer



def test_shift_vector_math():
    """Verify the vector shifting formula: User' + alpha * Repo, normalized to 1."""
    user_vec = [1.0, 0.0]
    repo_vec = [0.0, 1.0]
    alpha = 0.5  # shift coefficient

    updated = shift_vector(user_vec, repo_vec, alpha)

    # Manual calculation:
    # updated_unnorm = [1.0, 0.5]
    # norm = sqrt(1.0^2 + 0.5^2) = sqrt(1.25) = 1.11803
    # normalized = [1.0 / 1.11803, 0.5 / 1.11803] = [0.894427, 0.447213]
    
    assert len(updated) == 2
    assert pytest.approx(updated[0], rel=1e-5) == 0.894427
    assert pytest.approx(updated[1], rel=1e-5) == 0.447213

    # Normalized check (L2 norm should be exactly 1)
    norm = np.linalg.norm(updated)
    assert pytest.approx(norm, rel=1e-5) == 1.0


def test_shift_vector_negative():
    """Verify shifting away works for negative alpha (e.g. skip/ignore)."""
    user_vec = [1.0, 0.0]
    repo_vec = [0.0, 1.0]
    alpha = -0.5

    updated = shift_vector(user_vec, repo_vec, alpha)
    # updated_unnorm = [1.0, -0.5]
    # normalized = [1.0 / 1.11803, -0.5 / 1.11803]
    
    assert pytest.approx(updated[0], rel=1e-5) == 0.894427
    assert pytest.approx(updated[1], rel=1e-5) == -0.447213
    assert pytest.approx(np.linalg.norm(updated), rel=1e-5) == 1.0


@patch("feedback.event_handlers.PostgreSQLConnector")
@patch("feedback.event_handlers.QdrantClient")
def test_handler_like_event(mock_qdrant_cls, mock_db_cls):
    """Test that handle_feedback runs updates in Postgres and shifts in Qdrant."""
    mock_db = MagicMock()
    mock_db.enabled = True
    mock_db_cls.return_value = mock_db

    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant

    # Mock user retrieval from Qdrant
    mock_user_point = MagicMock()
    # Unnamed 384-dimensional user vector
    user_vec = [0.1] * 384
    mock_user_point.vector = user_vec
    mock_user_point.payload = {"user_id": "test_user", "skills": ["Python"]}
    mock_qdrant.retrieve.side_effect = [
        [mock_user_point],  # first call: user profile
        [MagicMock(vector=[0.2] * 384)]  # second call: repository
    ]

    handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    
    # Process like event
    success = handler.handle_feedback("test_user", "test-owner/test-repo", "like")
    
    assert success is True

    # Assert Postgres metric increment was executed
    assert mock_db.connect.call_count == 2  # 1 for metric, 1 for cache invalidation
    conn = mock_db.connect()
    cursor = conn.cursor()
    
    # Verify increment SQL was called
    args, _ = cursor.execute.call_args_list[0]
    sql = args[0]
    assert "UPDATE Repo" in sql
    assert "likes_count" in sql

    # Verify cache invalidation SQL was called
    args_cache, _ = cursor.execute.call_args_list[1]
    sql_cache = args_cache[0]
    assert "DELETE FROM user_recommendation_batches" in sql_cache

    # Assert Qdrant upsert was called with updated vector
    assert mock_qdrant.upsert.call_count == 1
    _, kwargs = mock_qdrant.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 1
    point = points[0]
    
    # Assert vector shifted: [0.1]*384 + 0.15 * [0.2]*384 = [0.13]*384, normalized
    expected_shifted = np.array([0.13] * 384)
    expected_shifted = (expected_shifted / np.linalg.norm(expected_shifted)).tolist()
    
    for val, exp in zip(point.vector, expected_shifted):
        assert pytest.approx(val, rel=1e-5) == exp


def test_api_feedback_submission():
    """Verify FastAPI handles request validation and returns HTTP 202."""
    client = TestClient(app)

    # Mock the producer to avoid hit Redis or async Queue queueing in testing
    with patch("api.main.producer") as mock_producer:
        mock_producer.submit_feedback = AsyncMock(return_value=True)

        # Test valid request
        response = client.post(
            "/api/v1/feedback",
            json={
                "user_id": "user_123",
                "repo_id": "facebook/react",
                "action": "like",
            },
        )
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        mock_producer.submit_feedback.assert_called_once_with(
            user_id="user_123",
            repo_id="facebook/react",
            action="like",
        )


def test_api_invalid_action():
    """Verify FastAPI rejects invalid actions with HTTP 400."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={
            "user_id": "user_123",
            "repo_id": "facebook/react",
            "action": "invalid_action",
        },
    )
    assert response.status_code == 400
    assert "Invalid action" in response.json()["detail"]
