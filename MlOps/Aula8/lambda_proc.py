"""
Simulating a slow processing function
"""
import time


def do_something(event, context):
    """This is the main function (handler)
    that will be called by AWS Lambda."""
    # Simulate slow processing
    time.sleep(5)
    return {
        "created_by": "Jean Pierre",
        "message": "data was processed",
    }