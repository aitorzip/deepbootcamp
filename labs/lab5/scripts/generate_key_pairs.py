#!/usr/bin/env python
from cloudexec import get_cloudexec_config, get_project_root
import boto3
import botocore.exceptions
import os

if __name__ == "__main__":
    config = get_cloudexec_config()

    key_names = dict()

    for region in config.aws_regions:
        ec2_client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_access_secret,
        )

        key_name = "{attendee_id}_{region}".format(
            attendee_id=config.attendee_id, region=region)

        key_names[region] = key_name

        print("Trying to create key pair with name %s" % key_name)
        import cloudexec
        file_name = cloudexec.local_ec2_key_pair_path(key_name)

        try:
            key_pair = ec2_client.create_key_pair(KeyName=key_name)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidKeyPair.Duplicate':
                if os.path.exists(file_name):
                    print("Key pair with name {key_name} already exists.".format(
                        key_name=key_name))
                else:
                    print(
                        "Key pair with name {key_name} exists remotely, but not locally! To fix this, "
                        "delete the remote one first".format(key_name=key_name))
                continue
            else:
                raise e

        print("Saving key pair file")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with os.fdopen(os.open(file_name, os.O_WRONLY | os.O_CREAT, 0o600), 'w') as handle:
            handle.write(key_pair['KeyMaterial'] + '\n')

    print("All set!")
    print("Now, edit your cloudexec.yml file, and update the `aws_key_pairs` entry to the following:")

    print()
    print("aws_key_pairs:")
    for region in config.aws_regions:
        print("    - {region}: {key_name}".format(region=region,
                                                  key_name=key_names[region]))
