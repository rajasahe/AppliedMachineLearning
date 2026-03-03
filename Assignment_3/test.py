"""
test.py - Unit tests for the score function and Flask integration tests.

Tests the score() function from score.py using the best model (LinearSVC)
saved during training experiments. Covers smoke tests, format tests,
sanity checks, edge cases, and typical input validation.

Also includes integration tests for the Flask /score endpoint.
"""

import os
import pickle
import signal
import subprocess
import sys
import time
import unittest

import requests
from sklearn.pipeline import Pipeline

from score import score


# ---------------------------------------------------------------------------
# Helper: load the saved model & vectorizer and combine into a Pipeline
# ---------------------------------------------------------------------------
def load_model():
    """Load the saved LinearSVC model and TF-IDF vectorizer as a Pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'best_model')

    with open(os.path.join(model_dir, 'linear_svc_model.pkl'), 'rb') as f:
        classifier = pickle.load(f)

    with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    # Combine into a sklearn Pipeline so score() can accept raw text
    model = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    return model


class TestScoreFunction(unittest.TestCase):
    """Unit tests for the score() function."""

    @classmethod
    def setUpClass(cls):
        """Load the model once for all tests."""
        cls.model = load_model()
        cls.default_threshold = 0.5

    # ------------------------------------------------------------------
    # 1. Smoke Test
    # ------------------------------------------------------------------
    def test_smoke(self):
        """Test that the function produces some output without crashing."""
        result = score("Hello, how are you?", self.model, self.default_threshold)
        self.assertIsNotNone(result)

    # ------------------------------------------------------------------
    # 2. Format Test - Output types
    # ------------------------------------------------------------------
    def test_output_is_tuple(self):
        """Test that the output is a tuple of (bool, float)."""
        result = score("Test message", self.model, self.default_threshold)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_prediction_type_is_bool(self):
        """Test that the prediction is a boolean."""
        prediction, _ = score("Test message", self.model, self.default_threshold)
        self.assertIsInstance(prediction, bool)

    def test_propensity_type_is_float(self):
        """Test that the propensity is a float."""
        _, propensity = score("Test message", self.model, self.default_threshold)
        self.assertIsInstance(propensity, float)

    # ------------------------------------------------------------------
    # 3. Sanity Check - Prediction value is 0 or 1 (True/False)
    # ------------------------------------------------------------------
    def test_prediction_is_binary(self):
        """Test that the prediction value is either True or False (0 or 1)."""
        prediction, _ = score("This is a sample text", self.model, self.default_threshold)
        self.assertIn(prediction, [True, False])

    # ------------------------------------------------------------------
    # 4. Sanity Check - Propensity score is between 0 and 1
    # ------------------------------------------------------------------
    def test_propensity_between_0_and_1(self):
        """Test that the propensity score is between 0 and 1."""
        _, propensity = score("Check this message", self.model, self.default_threshold)
        self.assertGreaterEqual(propensity, 0.0)
        self.assertLessEqual(propensity, 1.0)

    def test_propensity_range_on_multiple_inputs(self):
        """Test propensity is in [0, 1] across diverse inputs."""
        texts = [
            "Hi there!",
            "WINNER! You have been selected for a prize!",
            "Meeting at 3pm tomorrow",
            "Call now to claim your FREE holiday",
            "",  # empty string
        ]
        for text in texts:
            _, propensity = score(text, self.model, self.default_threshold)
            self.assertGreaterEqual(propensity, 0.0,
                                    f"Propensity {propensity} < 0 for text: '{text}'")
            self.assertLessEqual(propensity, 1.0,
                                 f"Propensity {propensity} > 1 for text: '{text}'")

    # ------------------------------------------------------------------
    # 5. Edge Case - Threshold 0 => prediction always True (1)
    # ------------------------------------------------------------------
    def test_threshold_zero_always_predicts_spam(self):
        """With threshold=0, propensity >= 0 is always true, so prediction should be True."""
        texts = [
            "Hello, how are you?",
            "Let's meet for lunch",
            "Can you send me the report?",
            "CONGRATULATIONS! You won a free trip!",
        ]
        for text in texts:
            prediction, propensity = score(text, self.model, threshold=0.0)
            self.assertTrue(prediction,
                            f"Expected prediction=True with threshold=0 for text: '{text}' "
                            f"(propensity={propensity})")

    # ------------------------------------------------------------------
    # 6. Edge Case - Threshold 1 => prediction always False (0)
    # ------------------------------------------------------------------
    def test_threshold_one_always_predicts_not_spam(self):
        """With threshold=1, propensity < 1 is expected (sigmoid never reaches exactly 1),
        so prediction should be False."""
        texts = [
            "Hello, how are you?",
            "Let's meet for lunch",
            "Can you send me the report?",
            "CONGRATULATIONS! You won a free trip!",
            "FREE cash prize call now 08001234567",
        ]
        for text in texts:
            prediction, propensity = score(text, self.model, threshold=1.0)
            self.assertFalse(prediction,
                             f"Expected prediction=False with threshold=1 for text: '{text}' "
                             f"(propensity={propensity})")

    # ------------------------------------------------------------------
    # 7. Typical Input - Obvious spam => prediction True (1)
    # ------------------------------------------------------------------
    def test_obvious_spam_detected(self):
        """Test that obvious spam messages are correctly identified."""
        spam_texts = [
            "WINNER!! You have been selected to receive a 900 prize reward! "
            "To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "Congratulations! You've won a free iPhone! Click here to claim your prize now!",
            "URGENT! You have won a 1 week FREE membership. Text WIN to 80086 to claim NOW!",
        ]
        for text in spam_texts:
            prediction, propensity = score(text, self.model, self.default_threshold)
            self.assertTrue(prediction,
                            f"Expected spam prediction for: '{text[:50]}...' "
                            f"(propensity={propensity:.4f})")

    # ------------------------------------------------------------------
    # 8. Typical Input - Obvious non-spam => prediction False (0)
    # ------------------------------------------------------------------
    def test_obvious_non_spam_detected(self):
        """Test that obvious non-spam (ham) messages are correctly identified."""
        ham_texts = [
            "Hey, are we still meeting for dinner tonight at 7?",
            "Can you pick up some groceries on your way home? We need milk and bread.",
            "Thanks for sending the notes from today's lecture. Really helpful!",
        ]
        for text in ham_texts:
            prediction, propensity = score(text, self.model, self.default_threshold)
            self.assertFalse(prediction,
                             f"Expected non-spam prediction for: '{text[:50]}...' "
                             f"(propensity={propensity:.4f})")


class TestFlaskApp(unittest.TestCase):
    """Integration tests for the Flask /score endpoint."""

    FLASK_PORT = 5000
    FLASK_URL = f'http://127.0.0.1:{FLASK_PORT}/score'
    flask_process = None

    @classmethod
    def setUpClass(cls):
        """Launch the Flask app as a subprocess before running tests."""
        cls.flask_process = subprocess.Popen(
            [sys.executable, 'app.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Windows: allows clean shutdown
        )

        # Wait for the Flask server to start up
        max_retries = 10
        for i in range(max_retries):
            try:
                requests.get(f'http://127.0.0.1:{cls.FLASK_PORT}/', timeout=1)
                break  # Server is up
            except requests.ConnectionError:
                time.sleep(1)
            except Exception:
                time.sleep(1)
        else:
            cls._shutdown_flask()
            raise RuntimeError('Flask app failed to start within timeout')

    @classmethod
    def tearDownClass(cls):
        """Shut down the Flask app after all tests."""
        cls._shutdown_flask()

    @classmethod
    def _shutdown_flask(cls):
        """Terminate the Flask subprocess."""
        if cls.flask_process is not None:
            cls.flask_process.terminate()
            try:
                cls.flask_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.flask_process.kill()
                cls.flask_process.wait()
            cls.flask_process = None

    # ------------------------------------------------------------------
    # Integration Tests
    # ------------------------------------------------------------------
    def test_flask_endpoint_returns_json(self):
        """Test that the /score endpoint returns a valid JSON response."""
        response = requests.post(
            self.FLASK_URL,
            json={'text': 'Hello, how are you?'},
            timeout=5
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('prediction', data)
        self.assertIn('propensity', data)

    def test_flask_prediction_type(self):
        """Test that prediction is a boolean in the JSON response."""
        response = requests.post(
            self.FLASK_URL,
            json={'text': 'Test message'},
            timeout=5
        )
        data = response.json()
        self.assertIsInstance(data['prediction'], bool)

    def test_flask_propensity_type_and_range(self):
        """Test that propensity is a float between 0 and 1."""
        response = requests.post(
            self.FLASK_URL,
            json={'text': 'Test message'},
            timeout=5
        )
        data = response.json()
        self.assertIsInstance(data['propensity'], float)
        self.assertGreaterEqual(data['propensity'], 0.0)
        self.assertLessEqual(data['propensity'], 1.0)

    def test_flask_spam_detection(self):
        """Test that the endpoint correctly identifies obvious spam."""
        response = requests.post(
            self.FLASK_URL,
            json={'text': 'WINNER!! You have been selected to receive a 900 prize reward! '
                          'To claim call 09061701461. Claim code KL341.'},
            timeout=5
        )
        data = response.json()
        self.assertTrue(data['prediction'], 'Expected spam to be detected')

    def test_flask_ham_detection(self):
        """Test that the endpoint correctly identifies obvious non-spam."""
        response = requests.post(
            self.FLASK_URL,
            json={'text': 'Hey, are we still meeting for dinner tonight at 7?'},
            timeout=5
        )
        data = response.json()
        self.assertFalse(data['prediction'], 'Expected ham to be detected')

    def test_flask_custom_threshold(self):
        """Test that the endpoint respects a custom threshold parameter."""
        # With threshold=0, everything should be spam
        response = requests.post(
            self.FLASK_URL,
            json={'text': 'Hello there', 'threshold': 0.0},
            timeout=5
        )
        data = response.json()
        self.assertTrue(data['prediction'],
                        'Expected prediction=True with threshold=0')

    def test_flask_missing_text_returns_400(self):
        """Test that missing 'text' field returns a 400 error."""
        response = requests.post(
            self.FLASK_URL,
            json={'wrong_field': 'some text'},
            timeout=5
        )
        self.assertEqual(response.status_code, 400)


def test_flask():
    """
    Standalone integration test for the Flask /score endpoint.

    Steps:
      1. Launch the Flask app as a background subprocess (via command line).
      2. Poll until the server is ready to accept connections.
      3. Send POST requests to /score and assert the response shape and values.
      4. Close the Flask app using os.system (taskkill on Windows).

    Raises:
        RuntimeError: If the Flask server doesn't start within the timeout.
        AssertionError: If any response check fails.
    """
    FLASK_PORT = 5000
    FLASK_URL = f'http://127.0.0.1:{FLASK_PORT}/score'
    BASE_URL  = f'http://127.0.0.1:{FLASK_PORT}/'
    base_dir  = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # 1. Launch Flask app using a subprocess (non-blocking background)
    # ------------------------------------------------------------------
    # CREATE_NEW_PROCESS_GROUP is Windows-only; use it only when on Windows
    if sys.platform == 'win32':
        flask_process = subprocess.Popen(
            [sys.executable, 'app.py'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        flask_process = subprocess.Popen(
            [sys.executable, 'app.py'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    # Poll until the server is up (up to 15 seconds)
    server_ready = False
    for _ in range(15):
        try:
            requests.get(BASE_URL, timeout=1)
            server_ready = True
            break
        except Exception:
            time.sleep(1)

    if not server_ready:
        os.system(f'taskkill /F /PID {flask_process.pid} > nul 2>&1')
        flask_process.wait()
        raise RuntimeError('Flask app failed to start within the 15-second timeout.')

    # ------------------------------------------------------------------
    # 2. Test the /score endpoint
    # ------------------------------------------------------------------
    try:
        # --- Basic response shape ---
        resp = requests.post(FLASK_URL, json={'text': 'Hello, how are you?'}, timeout=5)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert 'prediction' in data,  "Response missing 'prediction' key"
        assert 'propensity' in data,  "Response missing 'propensity' key"
        assert isinstance(data['prediction'], bool),  "prediction must be a bool"
        assert isinstance(data['propensity'], float), "propensity must be a float"
        assert 0.0 <= data['propensity'] <= 1.0, \
            f"propensity {data['propensity']} out of [0, 1] range"

        # --- Spam message should predict True ---
        resp_spam = requests.post(
            FLASK_URL,
            json={'text': 'WINNER!! You have been selected to receive a £900 prize reward! '
                          'To claim call 09061701461. Claim code KL341.'},
            timeout=5
        )
        assert resp_spam.status_code == 200
        assert resp_spam.json()['prediction'] is True, "Expected spam to be detected"

        # --- Ham message should predict False ---
        resp_ham = requests.post(
            FLASK_URL,
            json={'text': 'Hey, are we still meeting for dinner tonight at 7?'},
            timeout=5
        )
        assert resp_ham.status_code == 200
        assert resp_ham.json()['prediction'] is False, "Expected ham to be detected"

        # --- Missing 'text' field should return 400 ---
        resp_bad = requests.post(FLASK_URL, json={'wrong_field': 'oops'}, timeout=5)
        assert resp_bad.status_code == 400, \
            f"Expected 400 for missing 'text', got {resp_bad.status_code}"

        print("test_flask: all assertions passed [OK]")

    finally:
        # ------------------------------------------------------------------
        # 3. Close the Flask app using os.system (command-line kill)
        # ------------------------------------------------------------------
        os.system(f'taskkill /F /PID {flask_process.pid} > nul 2>&1')
        flask_process.wait()


if __name__ == '__main__':
    unittest.main()

