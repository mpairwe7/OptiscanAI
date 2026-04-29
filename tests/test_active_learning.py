"""Tests for ActiveLearningManager."""

import pytest
import tempfile
import shutil
import os

from src.active_learning.manager import (
    ActiveLearningManager,
    ReviewCase,
    ReviewStatus,
    SamplingStrategy,
)


@pytest.fixture
def manager():
    tmpdir = tempfile.mkdtemp(dir="/tmp")
    mgr = ActiveLearningManager(
        queue_dir=os.path.join(tmpdir, "al_queue"),
        confidence_threshold=0.65,
        uncertainty_threshold=0.3,
        retrain_threshold=5,  # Low threshold for testing
    )
    yield mgr
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestFlagging:
    def test_low_confidence_flagged(self, manager):
        preds = {"DR": 0.4, "ARMD": 0.3}
        uncertainty = {"epistemic": {"DR": 0.1, "ARMD": 0.1}}
        assert manager.should_flag_for_review(preds, uncertainty)

    def test_high_confidence_not_flagged(self, manager):
        preds = {"DR": 0.9, "ARMD": 0.1}
        uncertainty = {"epistemic": {"DR": 0.05, "ARMD": 0.05}}
        assert not manager.should_flag_for_review(preds, uncertainty)

    def test_high_uncertainty_flagged(self, manager):
        preds = {"DR": 0.8}
        uncertainty = {"epistemic": {"DR": 0.5}}
        assert manager.should_flag_for_review(preds, uncertainty)

    def test_clinical_priority_flagged(self, manager):
        """Sight-threatening conditions at moderate confidence should be flagged."""
        preds = {"DR": 0.5, "CRVO": 0.4}
        uncertainty = {"epistemic": {"DR": 0.1, "CRVO": 0.1}}
        assert manager.should_flag_for_review(preds, uncertainty)


class TestQueueManagement:
    def test_add_to_queue(self, manager):
        preds = {"DR": 0.3}
        uncertainty = {"epistemic": {"DR": 0.1}}
        case = manager.add_to_review_queue(
            case_id="test-001",
            image_path="/data/img001.jpg",
            predictions=preds,
            uncertainty=uncertainty,
            model_version="v1.0",
        )
        assert case is not None
        assert case.case_id == "test-001"
        assert len(manager.pending_queue) == 1

    def test_high_confidence_not_added(self, manager):
        preds = {"DR": 0.95}
        uncertainty = {"epistemic": {"DR": 0.01}}
        case = manager.add_to_review_queue(
            case_id="test-002",
            image_path="/data/img002.jpg",
            predictions=preds,
            uncertainty=uncertainty,
        )
        assert case is None
        assert len(manager.pending_queue) == 0

    def test_get_review_batch(self, manager):
        for i in range(5):
            manager.add_to_review_queue(
                case_id=f"case-{i}",
                image_path=f"/data/img{i}.jpg",
                predictions={"DR": 0.3 + i * 0.05},
                uncertainty={"epistemic": {"DR": 0.2}},
            )
        batch = manager.get_next_review_batch(batch_size=3)
        assert len(batch) == 3


class TestReviewSubmission:
    def test_approve_prediction(self, manager):
        manager.add_to_review_queue(
            case_id="review-001",
            image_path="/data/img.jpg",
            predictions={"DR": 0.4},
            uncertainty={"epistemic": {"DR": 0.2}},
        )
        success = manager.submit_review(
            case_id="review-001",
            reviewer_id="dr-smith",
            approved=True,
        )
        assert success
        assert len(manager.pending_queue) == 0
        assert len(manager.completed_reviews) == 1

    def test_correct_prediction(self, manager):
        manager.add_to_review_queue(
            case_id="review-002",
            image_path="/data/img.jpg",
            predictions={"DR": 0.4},
            uncertainty={"epistemic": {"DR": 0.2}},
        )
        success = manager.submit_review(
            case_id="review-002",
            reviewer_id="dr-jones",
            corrected_labels={"DR": 1.0, "CME": 0.8},
            notes="Clear diabetic retinopathy with macular edema",
        )
        assert success
        assert len(manager.fine_tune_queue) == 1

    def test_review_nonexistent_case(self, manager):
        success = manager.submit_review(
            case_id="nonexistent",
            reviewer_id="dr-smith",
            approved=True,
        )
        assert not success


class TestRetrainingTrigger:
    def test_retrain_triggered_at_threshold(self, manager):
        """Test that retraining is triggered when threshold is reached."""
        for i in range(5):
            manager.add_to_review_queue(
                case_id=f"rt-{i}",
                image_path=f"/data/img{i}.jpg",
                predictions={"DR": 0.3},
                uncertainty={"epistemic": {"DR": 0.2}},
            )
            manager.submit_review(
                case_id=f"rt-{i}",
                reviewer_id="dr-smith",
                corrected_labels={"DR": 1.0},
            )

        # After 5 corrections (threshold), fine_tune_queue should be cleared
        assert len(manager.fine_tune_queue) == 0
        assert manager.stats["retraining_triggers"] == 1


class TestStatistics:
    def test_statistics(self, manager):
        stats = manager.get_statistics()
        assert "pending_reviews" in stats
        assert "completed_reviews" in stats
        assert "fine_tune_queue" in stats
        assert "retrain_threshold" in stats
