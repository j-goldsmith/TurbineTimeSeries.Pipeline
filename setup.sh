pip install awscli

# Go to Home Directory
# Test if ~ works on windows
cd ~

# Create directory .aws if it does not exist 
mkdir -p .aws

cd .aws

# > operator means overwrite config file if exists
echo '[default]' > ~/.aws/config

# >> operator means append to end of file
echo 'output = json' >> ~/.aws/config

echo 'region = us-east-1' >> ~/.aws/config

## overwrite credentials file if exists
echo '[default]' > ~/.aws/credentials

echo 'aws_access_key_id = AKIAJLEI5OYOQA232WAQ' >> ~/.aws/credentials

echo 'aws_secret_access_key = +Wd0DSAMUyFh8F4fSCzG+m/DGU0kQEySNVmN6bIY' >> ~/.aws/credentials

echo -e '\n\nCheck your aws cli install with the command: aws s3 ls\n\n'

