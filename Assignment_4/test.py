import os
import subprocess
import time
import requests
import pytest

def test_docker():
    """
    Test the flask app running inside a Docker container.
    - Launches the docker container using commandline
    - Sends a request to the localhost endpoint /score
    - Checks if the response is as expected
    - Closes the docker container
    """
    image_name = "flask_spam_app"
    container_name = "test_flask_app_container"

    # Ensure any existing container is stopped and removed
    subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Build the docker image
    build_cmd = ["docker", "build", "-t", image_name, "."]
    subprocess.run(build_cmd, check=True)
    
    # Run the docker container
    run_cmd = ["docker", "run", "-d", "-p", "5000:5000", "--name", container_name, image_name]
    subprocess.run(run_cmd, check=True)
    
    try:
        url = "http://localhost:5000/score"
        started = False
        
        # Give the flask app a few seconds to start accepting requests
        for _ in range(15):
            try:
                response = requests.post(url, json={"text": "test"}, timeout=2)
                if response.status_code == 200:
                    started = True
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(2)
                continue
                
        assert started, "Flask app inside Docker container failed to start within the given time."
        
        # Test 1: Spam message
        sample_spam = "Congratulations! You have won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now."
        response_spam = requests.post(url, json={"text": sample_spam})
        assert response_spam.status_code == 200
        data_spam = response_spam.json()
        assert "prediction" in data_spam
        assert "propensity" in data_spam
        
        # Test 2: Ham message
        sample_ham = "Hey, are we still meeting for lunch at 1 PM?"
        response_ham = requests.post(url, json={"text": sample_ham})
        assert response_ham.status_code == 200
        data_ham = response_ham.json()
        assert "prediction" in data_ham
        assert "propensity" in data_ham
        
    finally:
        # Close the docker container
        subprocess.run(["docker", "stop", container_name], check=True)
        subprocess.run(["docker", "rm", container_name], check=True)
