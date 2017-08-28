#!/usr/bin/env python

import cloudexec
import os
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('--all', action='store_true', default=False)
    args = parser.parse_args()
    remote_dir = "s3://{bucket}/{bucket_root}/experiments".format(
        bucket=cloudexec.get_cloudexec_config().s3_bucket,
        bucket_root=cloudexec.get_cloudexec_config().s3_bucket_root
    )
    local_dir = os.path.join(cloudexec.get_project_root(), "data", "s3")
    if args.folder:
        remote_dir = os.path.join(remote_dir, args.folder)
        local_dir = os.path.join(local_dir, args.folder)
    s3_env = dict(
        os.environ,
        AWS_ACCESS_KEY_ID=cloudexec.get_cloudexec_config().aws_access_key,
        AWS_SECRET_ACCESS_KEY=cloudexec.get_cloudexec_config().aws_access_secret,
        AWS_REGION=cloudexec.get_cloudexec_config().aws_s3_region,
    )
    if not args.all:
        command = ("""
            aws s3 sync --exclude '*' {s3_periodic_sync_include_flags} --content-type "UTF-8" {remote_dir} {local_dir} 
        """.format(local_dir=local_dir, remote_dir=remote_dir,
                   s3_periodic_sync_include_flags=cloudexec.get_cloudexec_config().s3_periodic_sync_include_flags))
    else:
        command = ("""
            aws s3 sync --content-type "UTF-8" {remote_dir} {local_dir}
        """.format(local_dir=local_dir, remote_dir=remote_dir))
    subprocess.check_call(command, shell=True, env=s3_env)
