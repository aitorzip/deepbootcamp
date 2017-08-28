#!/usr/bin/env python


def main():
    import cloudexec
    import boto3
    import botocore.exceptions
    import os
    import subprocess
    config = cloudexec.get_cloudexec_config()

    assert len({
        config.attendee_id,
        config.ec2_instance_label,
        config.s3_bucket_root
    }) == 1, "attendee_id, ec2_instance_label, s3_bucket_root should have the same value"

    print("Testing attendee_id, aws_access_key, and aws_access_secret...")

    iam_client = boto3.client(
        "iam",
        region_name=config.aws_regions[0],
        aws_access_key_id=config.aws_access_key,
        aws_secret_access_key=config.aws_access_secret,
    )
    try:
        iam_client.list_access_keys(UserName=config.attendee_id)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'InvalidClientTokenId':
            print("aws_access_key is not set properly!")
            exit()
        elif e.response['Error']['Code'] == 'SignatureDoesNotMatch':
            print("aws_access_secret is not set properly!")
            exit()
        elif e.response['Error']['Code'] == 'AccessDenied':
            print("attendee_id is not set properly!")
            exit()
        else:
            raise e

    # Check if key pair exists

    for region in config.aws_regions:
        print("Checking key pair in region %s" % region)
        if region not in config.aws_key_pairs:
            print("Key pair in region %s is not set properly!" % region)
            exit()
        key_pair_name = config.aws_key_pairs[region]
        key_pair_path = cloudexec.local_ec2_key_pair_path(key_pair_name)
        if not os.path.exists(key_pair_path):
            print("Missing local key pair file at %s" % key_pair_path)
            exit()
        ec2_client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_access_secret,
        )
        try:
            response = ec2_client.describe_key_pairs(
                KeyNames=[config.aws_key_pairs[region]]
            )
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidKeyPair.NotFound':
                print("Key pair in region %s is not set properly!" % region)
                exit()
            else:
                raise e
        remote_fingerprint = response['KeyPairs'][0]['KeyFingerprint']

        # Get local key fingerprint

        ps = subprocess.Popen(
            ["openssl", "pkcs8", "-in", key_pair_path,
                "-nocrypt", "-topk8", "-outform", "DER"],
            stdout=subprocess.PIPE
        )
        local_fingerprint = subprocess.check_output(
            ["openssl", "sha1", "-c"], stdin=ps.stdout)
        # Strip irrelevant information
        local_fingerprint = local_fingerprint.decode().split('= ')[-1][:-1]

        if remote_fingerprint != local_fingerprint:
            print("Local key pair file does not match EC2 record!")
            exit()

    print("Your EC2 configuration has passed all checks!")


if __name__ == "__main__":
    main()
