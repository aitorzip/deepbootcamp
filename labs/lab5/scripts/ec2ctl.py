#!/usr/bin/env python
import datetime
import json
import logging
import multiprocessing
import os
import re
import sys

import click
import subprocess
import time

import boto3
from cloudexec import query_yes_no, get_cloudexec_config, get_project_root, local_ec2_key_pair_path
import numpy as np

DEBUG_LOGGING_MAP = {
    0: logging.CRITICAL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG
}


@click.group()
@click.option('--verbose', '-v',
              help="Sets the debug noise level, specify multiple times "
                   "for more verbosity.",
              type=click.IntRange(0, 3, clamp=True),
              count=True)
@click.pass_context
def cli(ctx, verbose):
    logger_handler = logging.StreamHandler(sys.stderr)
    logger_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(logger_handler)
    logging.getLogger().setLevel(DEBUG_LOGGING_MAP.get(verbose, logging.DEBUG))


def get_clients():
    clients = []
    config = get_cloudexec_config()
    regions = config.aws_regions
    for region in regions:
        client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_access_secret,
        )
        client.region = region
        clients.append(client)
    return clients


def _collect_instances(region):
    try:
        config = get_cloudexec_config()
        client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_access_secret,
        )
        print("Collecting instances in region", region)
        instances = sum([x['Instances'] for x in client.describe_instances(
            Filters=[
                {
                    'Name': 'instance-state-name',
                    'Values': [
                        'running'
                    ]
                }
            ]
        )['Reservations']], [])
        for instance in instances:
            instance['Region'] = region
        return instances
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


def get_tag_value(instance, tag_name):
    if 'Tags' in instance:
        try:
            tags = instance['Tags']
            name_tag = [t for t in tags if t['Key'] == tag_name][0]
            return name_tag['Value']
        except IndexError:
            return None
    return None


def get_all_instances():
    config = get_cloudexec_config()
    with multiprocessing.Pool(10) as pool:
        all_instances = sum(
            pool.map(_collect_instances, config.aws_regions), [])
    all_instances = [x for x in all_instances if get_tag_value(
        x, 'Owner') == config.attendee_id]
    return all_instances


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='verbose')
def jobs(verbose):
    jobs = []
    config = get_cloudexec_config()
    for instance in get_all_instances():
        exp_name = get_tag_value(instance, 'ExpName')
        if exp_name is None:
            exp_name = '(None)'
        if (exp_name is not None) and (instance['State']['Name'] != 'terminated'):
            jobs.append((exp_name, instance['Placement']['AvailabilityZone']))

    for job in sorted(jobs):
        print(*job)


@cli.command()
@click.argument('job')
def ssh(job):
    config = get_cloudexec_config()
    run_command = None
    for instance in get_all_instances():
        exp_name = get_tag_value(instance, 'ExpName')
        if exp_name == job:
            ip_addr = instance['PublicIpAddress']
            exp_group = get_tag_value(instance, 'ExpGroup')
            key_name = config.aws_key_pairs[instance['Region']]
            key_path = local_ec2_key_pair_path(key_name)
            ec2_path = os.path.join(
                config.ec2_project_root, "data/local/{exp_group}/{exp_name}".format(
                    exp_group=exp_group,
                    exp_name=exp_name
                )
            )
            command = " ".join([
                "ssh",
                "-oStrictHostKeyChecking=no",
                "-oConnectTimeout=10",
                "-i",
                key_path,
                "-t",
                "ubuntu@" + ip_addr,
                "'cd %s; exec bash -l'" % ec2_path
            ])
            print(command)

            def run_command():
                return os.system(command)
    if run_command is not None:
        while True:
            if run_command() == 0:
                break
            else:
                time.sleep(1)
                print("Retrying")
    else:
        print("Not found!")


def _copy_policy_params(job):
    config = get_cloudexec_config()
    for instance in get_all_instances():
        exp_name = get_tag_value(instance, 'ExpName')
        if exp_name == job:
            ip_addr = instance['PublicIpAddress']
            exp_group = get_tag_value(instance, 'ExpGroup')
            key_name = config.aws_key_pairs[instance['Region']]
            key_path = local_ec2_key_pair_path(key_name)
            remote_snapshots_path = os.path.join(
                config.ec2_project_root, "data/local/{exp_group}/{exp_name}/snapshots".format(
                    exp_group=exp_group,
                    exp_name=exp_name
                )
            )
            ssh_prefix = [
                "ssh",
                "-oStrictHostKeyChecking=no",
                "-oConnectTimeout=10",
                "-i",
                key_path,
                "ubuntu@{ip}".format(ip=ip_addr),
            ]

            ls_command = ssh_prefix + ["ls " + remote_snapshots_path]
            try:
                pkl_files = subprocess.check_output(ls_command)
            except subprocess.CalledProcessError:
                print("The snapshots folder does not exist yet. If the experiment is just launched, wait till the "
                      "first snapshot becomes available.")
                exit(0)

            pkl_files = pkl_files.decode().splitlines()

            if 'latest.pkl' in pkl_files:
                pkl_file = 'latest.pkl'
            else:
                pkl_file = sorted(
                    pkl_files, key=lambda x: int(x.split('.')[0]))[-1]

            remote_pkl_path = os.path.join(remote_snapshots_path, pkl_file)

            copy_command = [
                "ssh",
                "-oStrictHostKeyChecking=no",
                "-oConnectTimeout=10",
                "-i",
                key_path,
                "ubuntu@{ip}".format(ip=ip_addr),
                "cp {remote_path} /tmp/params.pkl".format(
                    remote_path=remote_pkl_path)
            ]
            print(" ".join(copy_command))
            subprocess.check_call(copy_command)
            local_exp_path = os.path.join(get_project_root(), "data/s3/{exp_group}/{exp_name}".format(
                exp_group=exp_group,
                exp_name=exp_name
            ))
            local_pkl_path = os.path.join(
                local_exp_path, "snapshots", pkl_file)
            os.makedirs(os.path.dirname(local_pkl_path), exist_ok=True)
            command = [
                "scp",
                "-oStrictHostKeyChecking=no",
                "-oConnectTimeout=10",
                "-i",
                key_path,
                "ubuntu@{ip}:/tmp/params.pkl".format(ip=ip_addr),
                local_pkl_path,
            ]
            print(" ".join(command))
            subprocess.check_call(command)
            return local_exp_path
    return False


@cli.command()
@click.argument('job')
def sim_policy(job):
    local_exp_path = _copy_policy_params(job)
    if local_exp_path:
        script = "scripts/sim_policy.py"
        command = [
            os.path.join(get_project_root(), script),
            local_exp_path,
        ]
        subprocess.check_call(command)
    else:
        print("Not found!")


@cli.command()
@click.argument('pattern')
def kill_f(pattern):
    print("trying to kill the pattern: ", pattern)
    to_kill = []
    to_kill_ids = {}
    for instance in get_all_instances():
        exp_name = get_tag_value(instance, 'ExpName')
        if exp_name is None or pattern in exp_name:
            instance_id = instance['InstanceId']
            region = instance['Region']
            if exp_name:
                if region not in to_kill_ids:
                    to_kill_ids[region] = []
                to_kill_ids[region].append(instance_id)
                to_kill.append(exp_name)

    print("This will kill the following jobs:")
    print(", ".join(sorted(to_kill)))
    if query_yes_no(question="Proceed?", default="no"):
        for client in get_clients():
            print("Terminating instances in region", client.region)
            ids = to_kill_ids.get(client.region, [])
            if len(ids) > 0:
                client.terminate_instances(
                    InstanceIds=to_kill_ids.get(client.region, [])
                )


@cli.command()
@click.argument('job')
def kill(job):
    to_kill = []
    to_kill_ids = {}
    for instance in get_all_instances():
        exp_name = get_tag_value(instance, 'ExpName')
        if exp_name == job:
            region = instance['Region']
            if region not in to_kill_ids:
                to_kill_ids[region] = []
            to_kill_ids[region].append(instance['InstanceId'])
            to_kill.append(exp_name)
            break

    print("This will kill the following jobs:")
    print(", ".join(sorted(to_kill)))
    if query_yes_no(question="Proceed?", default="no"):
        for client in get_clients():
            print("Terminating instances in region", client.region)
            ids = to_kill_ids.get(client.region, [])
            if len(ids) > 0:
                client.terminate_instances(
                    InstanceIds=to_kill_ids.get(client.region, [])
                )


def fetch_zone_prices(instance_type, zone, duration):
    clients = get_clients()
    for client in clients:
        if zone.startswith(client.region):

            all_prices = []
            all_ts = []
            for response in client.get_paginator('describe_spot_price_history').paginate(
                    InstanceTypes=[instance_type],
                    ProductDescriptions=['Linux/UNIX'],
                    AvailabilityZone=zone,
            ):
                history = response['SpotPriceHistory']
                prices = [float(x['SpotPrice']) for x in history]
                timestamps = [x['Timestamp'] for x in history]

                all_prices.extend(prices)
                all_ts.extend(timestamps)

                if len(all_ts) > 0:

                    delta = max(all_ts) - min(all_ts)
                    if delta.total_seconds() >= duration:
                        break

            return zone, all_prices, all_ts


def fetch_zones(region):
    clients = get_clients()
    for client in clients:
        if client.region == region:
            zones = [x['ZoneName'] for x in client.describe_availability_zones()[
                'AvailabilityZones']]
            return zones


@cli.command()
@click.argument('instance_type')
@click.option('--duration', '-d', help="Specify the duration to measure the maximum price. Defaults to 1 day. "
                                       "Examples: 100s, 1h, 2d, 1w", type=str, default='1d')
def spot_history(instance_type, duration):
    config = get_cloudexec_config()
    num_duration = int(duration[:-1])
    if re.match(r"^(\d+)d$", duration):
        duration = int(duration[:-1]) * 86400
        print("Querying maximum spot price in each zone within the past {duration} day(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)h$", duration):
        duration = int(duration[:-1]) * 3600
        print("Querying maximum spot price in each zone within the past {duration} hour(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)w$", duration):
        duration = int(duration[:-1]) * 86400 * 7
        print("Querying maximum spot price in each zone within the past {duration} week(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)m$", duration):
        duration = int(duration[:-1]) * 86400 * 30
        print("Querying maximum spot price in each zone within the past {duration} month(s)...".format(
            duration=num_duration))
    elif re.match(r"^(\d+)s$", duration):
        duration = int(duration[:-1])
        print("Querying maximum spot price in each zone within the past {duration} second(s)...".format(
            duration=num_duration))
    else:
        raise ValueError(
            "Unrecognized duration: {duration}".format(duration=duration))

    with multiprocessing.Pool(100) as pool:
        print('Fetching the list of all availability zones...')
        zones = sum(pool.starmap(
            fetch_zones, [(x,) for x in config.aws_regions]), [])
        print('Querying spot price in each zone...')
        results = pool.starmap(fetch_zone_prices, [(
            instance_type, zone, duration) for zone in zones])

        price_list = []

        for zone, prices, timestamps in results:
            if len(prices) > 0:
                sorted_prices_ts = sorted(
                    zip(prices, timestamps), key=lambda x: x[1])
                cur_time = datetime.datetime.now(
                    tz=sorted_prices_ts[0][1].tzinfo)
                sorted_prices, sorted_ts = [np.asarray(
                    x) for x in zip(*sorted_prices_ts)]
                cutoff = cur_time - datetime.timedelta(seconds=duration)

                valid_ids = np.where(np.asarray(sorted_ts) > cutoff)[0]
                if len(valid_ids) == 0:
                    first_idx = 0
                else:
                    first_idx = max(0, valid_ids[0] - 1)

                max_price = max(sorted_prices[first_idx:])

                price_list.append((zone, max_price))

        print("Spot pricing information for instance type {type}".format(
            type=instance_type))

        list_string = ''
        for zone, price in sorted(price_list, key=lambda x: x[1]):
            print("Zone: {zone}, Max Price: {price}".format(
                zone=zone, price=price))
            list_string += "'{}', ".format(zone)
        print(list_string)


@cli.command()
def ami():
    clients = get_clients()
    for client in clients:
        images = client.describe_images(Owners=['self'])['Images']
        for img in images:
            print('{name} in {region}'.format(
                name=img['Name'], region=client.region))


if __name__ == '__main__':
    cli()
