import os
import re

NUMA_PATH="/sys/devices/system/node"

def get_numa_info():
  ret = []
  if os.path.exists(NUMA_PATH):
    items = os.listdir(NUMA_PATH)
    nodes = list(filter(lambda str:str.startswith("node"), items))
    if nodes:
      for node in nodes:
        items = os.listdir(os.path.join(NUMA_PATH, node))
        cpus = [re.findall("cpu\d+", cpu) for cpu in items] 
        cpus = list(filter(lambda x:x, cpus))
        cpu_ids = [int(cpu[0].split('cpu')[1]) for cpu in cpus]
        cpu_ids = sorted(cpu_ids)
        ret.append(cpu_ids)
  else:
    print("NUMA PATH %s NOT FOUND" % NUMA_PATH)
  return ret

if __name__ == "__main__":
    print(get_numa_info())