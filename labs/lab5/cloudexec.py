import base64
import inspect
import random
import tempfile
from collections import defaultdict
import easydict
from copy import deepcopy
import logger
import os
import datetime
import subprocess
import cloudpickle
import click
import yaml
import dateutil.tz
import mako.template
import boto3
import botocore.exceptions
import numpy as np
import collections
import json
import sys


def serialize_call(target, variant):
    return base64.b64encode(cloudpickle.dumps(dict(target=target, variant=variant))).decode()


def deserialize_call(serialized):
    d = cloudpickle.loads(base64.b64decode(serialized))
    return d["target"], d["variant"]


def to_entrypoint_command(task, config):
    serialized = serialize_call(task.target, task.variant)
    if config.log_dir is None:
        config.log_dir = os.path.join(
            config.project_root, "data", "local", config.exp_group, config.exp_name)
    if config.entrypoint is None:
        config.entrypoint = os.path.join(config.project_root, "cloudexec.py")
    return [
        "python",
        config.entrypoint,
        serialized,
        "--logdir",
        config.log_dir
    ]


def to_docker_command(task, config):
    if config.docker_image is None:
        config.docker_image = get_cloudexec_config().docker_image

    # create config object for local
    project_root = config.project_root

    volumes = [(project_root, project_root)] + list(config.docker_volumes)

    if config.use_gpu:
        docker_command = ["nvidia-docker", "run", "-e", "USE_GPU=True"]
    else:
        docker_command = ["docker", "run"]

    # Add volumes
    for local_path, docker_path in volumes:
        docker_command += ["-v", "{local_path}:{docker_path}".format(
            local_path=local_path, docker_path=docker_path)]
    # Configure docker image
    docker_command += ["-i", config.docker_image]
    docker_command += to_entrypoint_command(task, config)

    return docker_command


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))


_cloudexec_config = None


def get_cloudexec_config():
    global _cloudexec_config
    if _cloudexec_config is None:
        cfg_file = os.path.join(get_project_root(), "cloudexec.yml")
        with open(cfg_file, "r") as f:
            config = yaml.load(f)
            for k, v in config.items():
                # quick hack to support environment variables
                if isinstance(v, str):
                    if v.startswith("ENV[\"") and v.endswith("\"]"):
                        config[k] = os.environ.get(
                            v[len("ENV[\""):-len("\"]")], "")
            _cloudexec_config = easydict.EasyDict(config)
    return _cloudexec_config


def local_mode(task, config):
    # configure project root to be the same
    if config.project_root is None:
        config.project_root = get_project_root()
    command = to_entrypoint_command(task, config)
    subprocess.call(command)


def local_docker_mode(task, config):
    # configure project root to be the same
    if config.project_root is None:
        config.project_root = get_project_root()
    command = to_docker_command(task, config)
    subprocess.call(command)


_exp_count = defaultdict(int)
_timestamp = datetime.datetime.now(
    dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

LAUNCHER_TEMPLATE = mako.template.Template("""
#!/bin/bash
{
    die() { status=$1; shift; echo "FATAL: $*"; exit $status; }

    # Get instance id
    EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`"

    echo "Launching docker service"
    service docker start

    echo "Start executing job"

    echo "Adding tags"
    while /bin/true; do
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value=${cloudexec_config.ec2_instance_label}/${config.exp_name} && \
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Label,Value=${cloudexec_config.ec2_instance_label} && \
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=ExpGroup,Value=${config.exp_group} && \
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=ExpName,Value=${config.exp_name} && \
        break
        sleep 5
    done
        

    # If docker credentials are provided, sign in first
    % if cloudexec_config.docker_username is not None:
    docker login -u ${cloudexec_config.docker_username} -p ${cloudexec_config.docker_password} ${cloudexec_config.docker_host}
    % endif

    echo "Pulling docker image"
    docker pull ${config.docker_image}

    echo "Downloading code from S3"
    aws s3 cp ${s3_code_path} /tmp/cloudexec_code.tar.gz

    echo "Unpacking downloaded code"
    mkdir -p ${config.project_root}
    tar -zxvf /tmp/cloudexec_code.tar.gz -C ${config.project_root}
    
    setfacl -dRm u:${cloudexec_config.ec2_user}:rwX ${config.project_root}
    setfacl -Rm u:${cloudexec_config.ec2_user}:rwX ${config.project_root}
 
    cd ${config.project_root}

    mkdir -p ${config.log_dir}
   
    % if cloudexec_config.s3_periodic_sync_interval > 0:
    while /bin/true; do
        aws s3 sync --exclude '*' ${cloudexec_config.s3_periodic_sync_include_flags} ${config.log_dir} ${s3_log_dir}
        sleep ${cloudexec_config.s3_periodic_sync_interval}
    done & echo "Periodic sync initiated"
    % endif

    while /bin/true; do
        if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
        then
            logger "Running shutdown hook."
            aws s3 cp /home/ubuntu/user_data.log ${s3_log_dir}/stdout.log
            aws s3 cp --recursive ${config.log_dir} ${s3_log_dir}
            break
        else
            # Spot instance not yet marked for termination.
            sleep 5
        fi
    done & echo "Termination sync hook launched"
    
    echo "Launching docker command"

    ${' '.join(docker_command)}

    aws s3 cp --recursive ${config.log_dir} ${s3_log_dir}

    aws s3 cp /home/ubuntu/user_data.log ${s3_log_dir}/stdout.log

    % if cloudexec_config.ec2_terminate_machine:
    # Keep trying to terminate machine until success
    while /bin/true; do
        EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`"
        aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID
        sleep 5
    done
    % endif

} >> /home/ubuntu/user_data.log 2>&1
""")


def local_ec2_key_pair_path(key_name):
    return os.path.join(get_project_root(), "private", "key_pairs", key_name + ".pem")


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


_cached_spot_prices = dict()


def get_cached_spot_prices(instance_type, regions, max_spot_price):
    result_list = []
    for region in regions:
        client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=get_cloudexec_config().aws_access_key,
            aws_secret_access_key=get_cloudexec_config().aws_access_secret,
        )
        subnets = get_cloudexec_config().aws_subnets[region]
        for zone in subnets.keys():
            key = (region, zone, instance_type)
            if key not in _cached_spot_prices:
                _cached_spot_prices[key] = client.describe_spot_price_history(
                    MaxResults=1,
                    InstanceTypes=[instance_type],
                    AvailabilityZone=zone,
                    ProductDescriptions=['Linux/UNIX']
                )['SpotPriceHistory'][0]['SpotPrice']
            spot_price = _cached_spot_prices[key]
            if float(spot_price) < float(max_spot_price):
                result_list.append((region, subnets[zone], spot_price))
    return result_list


_cached_s3_code_path = None
_ec2_confirmed = False


def ec2_mode(task, config):
    global _ec2_confirmed
    if not _ec2_confirmed:
        _ec2_confirmed = query_yes_no("Running in EC2 mode. Confirm?")
        if not _ec2_confirmed:
            exit()

    s3_env = dict(
        os.environ,
        AWS_ACCESS_KEY_ID=get_cloudexec_config().aws_access_key,
        AWS_SECRET_ACCESS_KEY=get_cloudexec_config().aws_access_secret,
        AWS_REGION=get_cloudexec_config().aws_s3_region,
    )
    s3_root_path = "s3://{bucket}/{bucket_root}/".format(
        bucket=get_cloudexec_config().s3_bucket,
        bucket_root=get_cloudexec_config().s3_bucket_root
    )

    # Set configurations

    if config.aws_instance_type is None:
        config.aws_instance_type = get_cloudexec_config().aws_instance_type

    if config.aws_use_spot_instances is None:
        config.aws_use_spot_instances = get_cloudexec_config().aws_use_spot_instances

    if config.aws_spot_price is None:
        config.aws_spot_price = get_cloudexec_config().aws_spot_price

    if config.aws_region is None or config.aws_subnet_id is None:
        if config.aws_region is None:
            regions = get_cloudexec_config().aws_regions
        else:
            regions = [config.aws_region]
        if config.aws_use_spot_instances:
            # Get real time spot prices, and only sample from regions where the spot price is higher than the current
            # price
            spot_prices = get_cached_spot_prices(
                instance_type=config.aws_instance_type,
                regions=regions,
                max_spot_price=config.aws_spot_price,
            )

            if len(spot_prices) == 0:
                raise ValueError("All availability zones within specification have spot price higher than the "
                                 "specified price (%s). Consider raising your spot price, switching to other regions,"
                                 "or using regular instances" % config.aws_spot_price)

            config.aws_region, config.aws_subnet_id, _ = random.choice(
                spot_prices)
        else:
            config.aws_region = random.choice(regions)
            if config.aws_subnet_id is None:
                all_subnets = list(get_cloudexec_config(
                ).aws_subnets[config.aws_region].values())
                config.aws_subnet_id = random.choice(all_subnets)

    if config.aws_region not in get_cloudexec_config().aws_key_pairs:
        raise ValueError(
            "Missing key pair for region {}!".format(config.aws_region))

    key_name = get_cloudexec_config().aws_key_pairs[config.aws_region]

    if not os.path.exists(local_ec2_key_pair_path(key_name)):
        raise ValueError("Key pair file missing!")

    # Compress all project code into a tar file
    global _cached_s3_code_path
    if _cached_s3_code_path is None:
        local_code_path = "/tmp/{}.tar.gz".format(_timestamp)
        tar_cmd = ["tar", "-zcf", local_code_path, "-C", get_project_root()]
        for pattern in get_cloudexec_config().s3_code_sync_ignores:
            tar_cmd += ["--exclude", pattern]
        tar_cmd += ["-h", "."]
        subprocess.check_call(tar_cmd, env=s3_env)

        # Upload code to S3
        s3_code_path = os.path.join(
            s3_root_path, "code", "{}.tar.gz".format(_timestamp))
        code_upload_cmd = ["aws", "s3", "cp", local_code_path, s3_code_path]
        subprocess.check_call(code_upload_cmd, env=s3_env)
        _cached_s3_code_path = s3_code_path
    else:
        s3_code_path = _cached_s3_code_path

    # Form bash script to be executed on S3

    if config.project_root is None:
        config.project_root = get_cloudexec_config().ec2_project_root

    docker_command = to_docker_command(
        task=task,
        config=config,
    )

    s3_log_dir = os.path.join(
        s3_root_path, "experiments", config.exp_group, config.exp_name)
    script = LAUNCHER_TEMPLATE.render(
        docker_command=docker_command,
        config=config,
        cloudexec_config=get_cloudexec_config(),
        s3_code_path=s3_code_path,
        s3_log_dir=s3_log_dir,
    )

    if get_cloudexec_config().debug:
        print(script)

    # Upload launcher script to S3
    with tempfile.NamedTemporaryFile() as f:
        f.write(script.encode())
        f.flush()
        local_script_path = f.name
        s3_script_path = os.path.join(
            s3_root_path, "scripts", "{}.sh".format(config.exp_name))
        script_upload_cmd = ["aws", "s3", "cp",
                             local_script_path, s3_script_path]
        subprocess.check_call(script_upload_cmd, env=s3_env)

    user_data = mako.template.Template("""#!/bin/bash
{
    export AWS_REGION=${aws_region}
    export AWS_DEFAULT_REGION=${aws_region}
    export AWS_ACCESS_KEY_ID=${aws_access_key}
    export AWS_SECRET_ACCESS_KEY=${aws_access_secret}
    
    # Wait until got instance id
    while /bin/true; do
        export EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`" && break
        sleep 5
    done
    
    echo "Instance ID: $EC2_INSTANCE_ID"
    
    # Wait until instance is tagged
    while /bin/true; do
        tag_output=`aws ec2 describe-tags --filters "Name=resource-id,Values=$EC2_INSTANCE_ID" "Name=key,Values=OwnerId" --region $AWS_REGION`
        n_tags=`echo $tag_output | python -c "import sys, json; print(len(json.load(sys.stdin)['Tags']))"`
        if [ "$n_tags" -gt "0" ]; then
            break
        fi
        sleep 5
    done
    
    echo "Instance tagged!"
    echo $tag_output
    
    setfacl -dRm u:${cloudexec_config.ec2_user}:rwX /home/ubuntu
    setfacl -Rm u:${cloudexec_config.ec2_user}:rwX /home/ubuntu

    aws s3 cp ${s3_script_path} /home/ubuntu/remote_script.sh
    chmod +x /home/ubuntu/remote_script.sh
    bash /home/ubuntu/remote_script.sh
    
    % if cloudexec_config.ec2_terminate_machine:
    # Keep trying to terminate machine until success
    while /bin/true; do
        aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID
        sleep 5
    done
    % endif

} >> /home/ubuntu/user_data.log 2>&1""") \
        .render(
        s3_script_path=s3_script_path,
        aws_region=config.aws_region,
        aws_access_key=get_cloudexec_config().aws_access_key,
        aws_access_secret=get_cloudexec_config().aws_access_secret,
        cloudexec_config=get_cloudexec_config(),
    )

    if get_cloudexec_config().debug:
        print(user_data)

    launch_specification = dict(
        ImageId=get_cloudexec_config().aws_image_id[config.aws_region],
        KeyName=key_name,
        UserData=base64.b64encode(user_data.encode()).decode("utf-8"),
        InstanceType=config.aws_instance_type,
        EbsOptimized=True,
        NetworkInterfaces=[
            dict(
                SubnetId=config.aws_subnet_id,
                Groups=[get_cloudexec_config(
                ).aws_security_groups[config.aws_region]],
                DeviceIndex=0,
                AssociatePublicIpAddress=True,
            )
        ],
        # IamInstanceProfile=dict(
        #     Name=get_cloudexec_config().aws_iam_instance_profile,
        # ),
        **(config.aws_extra_launch_specification or {})
    )

    client = boto3.client(
        "ec2",
        region_name=config.aws_region,
        aws_access_key_id=get_cloudexec_config().aws_access_key,
        aws_secret_access_key=get_cloudexec_config().aws_access_secret,
    )

    if config.aws_use_spot_instances:
        valid_from = datetime.datetime.utcnow()
        valid_from = valid_from.replace(
            microsecond=0) + datetime.timedelta(seconds=3)
        valid_until = valid_from + datetime.timedelta(hours=1)
        response = client.request_spot_instances(
            InstanceCount=1,
            LaunchSpecification=launch_specification,
            SpotPrice=str(config.aws_spot_price),
            # ValidFrom=valid_from,
            # ValidUntil=valid_until,
        )
        spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        for _ in range(10):
            try:
                client.create_tags(
                    Resources=[spot_request_id],
                    Tags=[{'Key': 'Name', 'Value': config.exp_name}],
                )
                break
            except botocore.exceptions.ClientError:
                continue
    else:
        launch_specification["MinCount"] = 1
        launch_specification["MaxCount"] = 1

        response = client.create_instances(
            **launch_specification
        )

    if get_cloudexec_config().debug:
        print(response)

    subnets = list(get_cloudexec_config(
    ).aws_subnets[config.aws_region].items())
    zone = [zone for zone, subnet in subnets if subnet == config.aws_subnet_id][0]

    print("Submitted job %s to EC2 in region %s (zone %s)" %
          (config.exp_name, config.aws_region, zone))


class Config(object):
    def __init__(
            self,
            exp_group="experiment",
            exp_name=None,
            project_root=None,
            log_dir=None,
            entrypoint=None,
            use_gpu=False,
            docker_image=None,
            docker_volumes=(),
            aws_region=None,
            aws_subnet_id=None,
            aws_use_spot_instances=None,
            aws_spot_price=None,
            aws_instance_type=None,
            aws_extra_launch_specification=None,
    ):
        self.exp_group = exp_group
        self.exp_name = exp_name
        self.project_root = project_root
        self.log_dir = log_dir
        self.entrypoint = entrypoint
        self.use_gpu = use_gpu
        self.docker_image = docker_image
        self.docker_volumes = docker_volumes
        self.aws_region = aws_region
        self.aws_subnet_id = aws_subnet_id
        self.aws_use_spot_instances = aws_use_spot_instances
        self.aws_spot_price = aws_spot_price
        self.aws_instance_type = aws_instance_type
        self.aws_extra_launch_specification = aws_extra_launch_specification


class Task(object):
    def __init__(self, target, variant):
        self.target = target
        self.variant = variant


def remote_call(task, config=None, mode=local_mode):
    if config is None:
        config = Config()
    else:
        config = deepcopy(config)
    if config.exp_name is None:
        _exp_count[config.exp_group] += 1
        config.exp_name = "%s_%s_%04d" % (
            config.exp_group, _timestamp, _exp_count[config.exp_group])
    mode(task, config)


class VariantGenerator(object):
    """
    Usage:

    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", ['x', 'y'])
    vg.variants() => # all combinations of [1,2,3] x ['x','y']

    Supports non-cyclic dependency among parameters:
    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", lambda param1: [param1+1, param1+2])
    vg.variants() => # ..
    """

    def __init__(self):
        self._variants = []
        self._populate_variants()

    def add(self, key, vals, **kwargs):
        self._variants.append((key, vals, kwargs))

    def _populate_variants(self):
        methods = inspect.getmembers(
            self.__class__, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
        methods = [x[1].__get__(self, self.__class__)
                   for x in methods if getattr(x[1], '__is_variant', False)]
        for m in methods:
            self.add(m.__name__, m, **getattr(m, "__variant_config", dict()))

    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        logger.info("Generating {} variants".format(len(ret)))
        return ret

    def ivariants(self):
        dependencies = list()
        for key, vals, _ in self._variants:
            if hasattr(vals, "__call__"):
                args = inspect.getfullargspec(vals).args
                if hasattr(vals, 'im_self') or hasattr(vals, "__self__"):
                    # remove the first 'self' parameter
                    args = args[1:]
                dependencies.append((key, set(args)))
            else:
                dependencies.append((key, set()))
        sorted_keys = []
        # topo sort all nodes
        while len(sorted_keys) < len(self._variants):
            # get all nodes with zero in-degree
            free_nodes = [k for k, v in dependencies if len(v) == 0]
            if len(free_nodes) == 0:
                error_msg = "Invalid parameter dependency: \n"
                for k, v in dependencies:
                    if len(v) > 0:
                        error_msg += k + " depends on " + " & ".join(v) + "\n"
                raise ValueError(error_msg)
            dependencies = [(k, v)
                            for k, v in dependencies if k not in free_nodes]
            # remove the free nodes from the remaining dependencies
            for _, v in dependencies:
                v.difference_update(free_nodes)
            sorted_keys += free_nodes
        return self._ivariants_sorted(sorted_keys)

    def _ivariants_sorted(self, sorted_keys):
        if len(sorted_keys) == 0:
            yield dict()
        else:
            first_keys = sorted_keys[:-1]
            first_variants = self._ivariants_sorted(first_keys)
            last_key = sorted_keys[-1]
            last_vals = [v for k, v, _ in self._variants if k == last_key][0]
            if hasattr(last_vals, "__call__"):
                last_val_keys = inspect.getfullargspec(last_vals).args
                if hasattr(last_vals, 'im_self') or hasattr(last_vals, '__self__'):
                    last_val_keys = last_val_keys[1:]
            else:
                last_val_keys = None
            for variant in first_variants:
                if hasattr(last_vals, "__call__"):
                    last_variants = last_vals(
                        **{k: variant[k] for k in last_val_keys})
                    for last_choice in last_variants:
                        yield easydict.EasyDict(variant, **{last_key: last_choice})
                else:
                    for last_choice in last_vals:
                        yield easydict.EasyDict(variant, **{last_key: last_choice})


def variant(*args, **kwargs):
    def _variant(fn):
        fn.__is_variant = True
        fn.__variant_config = kwargs
        return fn

    if len(args) == 1 and isinstance(args[0], collections.Callable):
        return _variant(args[0])
    return _variant


@click.command()
@click.argument("serialized_call", type=str)
@click.option("--logdir", type=str, help="directory used for logging", required=True)
def entrypoint(serialized_call, logdir):
    """
    Entry point for delayed execution.

    Usage (usually won't be directly called from a shell, but instead programmatically launched):

    python cloudexec.py <serialized_call>

    where serialized_call is a base64-encoded pickled function call.
    """
    target, variant = deserialize_call(serialized_call)
    with logger.session(dir=logdir):
        # save parameter file
        with open(os.path.join(logdir, "variant.json"), "w") as f:
            f.write(json.dumps(variant))
        target(variant)


if __name__ == "__main__":
    entrypoint()

__all__ = [
    "variant",
    "VariantGenerator",
    "remote_call",
    "Config",
    "Task",
    "local_mode",
    "local_docker_mode",
    "ec2_mode"
]
