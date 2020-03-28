#!/usr/bin/python

from __future__ import print_function

import os
import re
import subprocess
import sys
import threading
import time
from functools import reduce

COMMON_REQUIRED_ENVS = ["DMLC_ROLE", "DMLC_NUM_WORKER", "DMLC_NUM_SERVER",
                        "DMLC_PS_ROOT_URI", "DMLC_PS_ROOT_PORT"]
WORKER_REQUIRED_ENVS = ["DMLC_WORKER_ID"]
NUMA_PATH = "/sys/devices/system/node"


def check_env():
    assert "DMLC_ROLE" in os.environ and \
           os.environ["DMLC_ROLE"].lower() in ["worker", "server", "scheduler"]
    required_envs = COMMON_REQUIRED_ENVS
    if os.environ["DMLC_ROLE"] == "worker":
        assert "DMLC_NUM_WORKER" in os.environ
        num_worker = int(os.environ["DMLC_NUM_WORKER"])
        assert num_worker >= 1
        if num_worker == 1:
            required_envs = []
        required_envs += WORKER_REQUIRED_ENVS
    for env in required_envs:
        if env not in os.environ:
            print("The env " + env + " is missing")
            os._exit(0)


def get_numa_info():
    ret = []
    if os.path.exists(NUMA_PATH):
        items = os.listdir(NUMA_PATH)
        nodes = list(filter(lambda str: str.startswith("node"), items))
        if nodes:
            for node in nodes:
                items = os.listdir(os.path.join(NUMA_PATH, node))
                cpus = [re.findall("cpu\d+", cpu) for cpu in items]
                cpus = list(filter(lambda x: x, cpus))
                cpu_ids = [int(cpu[0].split('cpu')[1]) for cpu in cpus]
                cpu_ids = sorted(cpu_ids)
                ret.append(cpu_ids)
    else:
        print("NUMA PATH %s NOT FOUND" % NUMA_PATH)
    return ret


def allocate_cpu(local_size):
    def _get_allocation(nodes, quota):
        if quota < 1:
            raise ValueError("quota should be no less than 1")
        ret = []
        for node in nodes:
            if len(node) < quota:
                continue
            split_index = []
            for i in range(1, quota):
                if node[i] != node[i-1] + 1:
                    split_index.append(i)
            last_idx = 0
            for idx in split_index:
                ret.append(node[last_idx:idx])
                quota -= idx - last_idx
                last_idx = idx
            ret.append(node[last_idx:last_idx+quota])
            for idx in sorted(range(quota), reverse=True):
                del node[idx]
            return ret
        return ret

    def _get_quota(nodes, local_size):
        cpu_nums = reduce(lambda x, y: (len(x) + len(y)), nodes)
        default_quota = 4
        while default_quota >= 1 and default_quota * local_size > cpu_nums:
            default_quota //= 2
        root_quota = cpu_nums - default_quota * (local_size - 1)
        node_size = len(nodes[0])
        while root_quota >= 1 and root_quota > node_size:
            root_quota //= 2
        return [default_quota] * (local_size - 1) + [root_quota]
    nodes = [list(range(0, 16)) + list(range(32, 48)),
             list(range(16, 32)) + list(range(48, 64))]
    quota_list = _get_quota(nodes, local_size)
    ret = []
    for quota in quota_list:
        ret.append(_get_allocation(nodes, quota))
    return ret


def worker(local_rank, local_size, command, allocation):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)
    if int(os.getenv("BYTEPS_ENABLE_GDB", 0)):
        if command.find("python") != 0:
            command = "python " + command
        command = "gdb -ex 'run' -ex 'bt' -batch --args " + command

    if local_rank == local_size - 1:
        numa = "OMP_NUM_THREADS=8 numactl --physcpubind "
        for cpu_set in allocation:
            numa += "{}-{},".format(cpu_set[0], cpu_set[-1])
        numa = numa.strip(',') + ' '
        command = numa + command
    else:
        numa = "numactl --physcpubind "
        for cpu_set in allocation:
            numa += "{}-{},".format(cpu_set[0], cpu_set[-1])
        numa = numa.strip(',') + ' '
        command = numa + command

    if os.environ.get("BYTEPS_TRACE_ON", "") == "1":
        print("\n!!!Enable profiling for WORKER_ID: %s and local_rank: %d!!!" %
              (os.environ.get("DMLC_WORKER_ID"), local_rank))
        print("BYTEPS_TRACE_START_STEP: %s\tBYTEPS_TRACE_END_STEP: %s\t BYTEPS_TRACE_DIR: %s" % (os.environ.get(
            "BYTEPS_TRACE_START_STEP", ""), os.environ.get("BYTEPS_TRACE_END_STEP", ""), os.environ.get("BYTEPS_TRACE_DIR", "")))
        print("Command: %s\n" % command)
        sys.stdout.flush()
        trace_path = os.path.join(os.environ.get(
            "BYTEPS_TRACE_DIR", "."), str(local_rank))
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)
    subprocess.check_call(command, env=my_env,
                          stdout=sys.stdout, stderr=sys.stderr, shell=True)


def launch_bps():
    print("BytePS launching " + os.environ["DMLC_ROLE"])
    sys.stdout.flush()
    check_env()
    if os.environ["DMLC_ROLE"] == "worker":
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            local_size = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
        else:
            local_size = 1
        t = [None] * local_size

        allocations = allocate_cpu(local_size)
        for i in range(local_size):
            command = ' '.join(sys.argv[1:])
            t[i] = threading.Thread(target=worker, args=[
                                    i, local_size, command, allocations[i]])
            t[i].daemon = True
            t[i].start()

        for i in range(local_size):
            t[i].join()

    else:
        import byteps.server


if __name__ == "__main__":
    launch_bps()
