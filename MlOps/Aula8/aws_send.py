import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# Lambda function name: do_something_concurrent_<YOUR_INSPER_USERNAME>
function_name = "do_something_concurrent_joaoprs3"
# Timeout in seconds. Default is 3.
timeout = 15
# Lambda basic execution role
lambda_role_arn = os.getenv("AWS_LAMBDA_ROLE_ARN")

# Create a Boto3 client for AWS Lambdac
session = boto3.Session(
    profile_name=os.getenv("AWS_PROFILE"), region_name=os.getenv("AWS_REGION")
)

lambda_client = session.client("lambda")

# Read the contents of the zip file that you want to deploy
with open("lambda_proc.zip", "rb") as f:
    zip_to_deploy = f.read()

lambda_response = lambda_client.create_function(
    FunctionName=function_name,
    Runtime="python3.10",
    Role=lambda_role_arn,
    Handler="lambda_proc.do_something",
    Code={"ZipFile": zip_to_deploy},
    Timeout=timeout,
)

print("Function ARN:", lambda_response["FunctionArn"])