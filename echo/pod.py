#!/usr/bin/env python3
"""RunPod pods for the echo/rank experiments (adapted from nd-rl's autoresearch/pod.py).

  python echo/pod.py up NAME [DISK_GB]   rent one GPU (GPUS order, community then secure), wait for ssh,
                                         upload echo/ + subliminal_pilot/ to /root/lp; state in echo/pods/NAME.json
  python echo/pod.py sh NAME 'CMD'       run CMD on the pod (bash -lc, cwd /root/lp)
  python echo/pod.py bg NAME LOG 'CMD'   run CMD detached (nohup setsid), output to /root/LOG
  python echo/pod.py push NAME           re-upload the code
  python echo/pod.py pull NAME REMOTE LOCAL
  python echo/pod.py down NAME           terminate (prints minutes and approximate cost)
  python echo/pod.py list
"""
import json, os, shlex, subprocess, sys, time, tomllib, urllib.request

API = "https://api.runpod.io/graphql"
KEY = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))["apikey"]
IMAGE = "runpod/pytorch:1.3.3-cu1290-torch291-ubuntu2204"
GPUS = os.environ.get("GPUS", "NVIDIA H100 NVL,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe").split(",")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE = f"{HERE}/pods"


def gql(q):
    req = urllib.request.Request(f"{API}?api_key={KEY}", method="POST", data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"})
    for i in range(4):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=60))
            break
        except (OSError, json.JSONDecodeError):
            time.sleep(10 * (i + 1))
    if "errors" in out:
        raise RuntimeError(json.dumps(out["errors"]))
    return out["data"]


def create(name, disk):
    pub = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()
    for cloud in ("COMMUNITY", "SECURE"):
        for gpu in GPUS:
            q = ("mutation { podFindAndDeployOnDemand(input: {"
                 f'cloudType: {cloud}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: {disk}, minVcpuCount: 6, minMemoryInGb: 48, '
                 f'gpuTypeId: "{gpu}", name: "{name}", imageName: "{IMAGE}", ports: "22/tcp", volumeMountPath: "/workspace", '
                 f'supportPublicIp: true, env: [{{key: "PUBLIC_KEY", value: "{pub}"}}]'
                 "}) { id costPerHr machine { gpuDisplayName } } }")
            try:
                pod = gql(q)["podFindAndDeployOnDemand"]
            except RuntimeError as e:
                if any(s in str(e) for s in ("SUPPLY_CONSTRAINT", "does not have the resources", "INTERNAL_SERVER_ERROR")):
                    continue
                raise
            if pod:
                return pod | {"gpu": gpu, "cloud": cloud}
    return None


def wait_ssh(pod_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rt = gql(f'query {{ pod(input: {{podId: "{pod_id}"}}) {{ runtime {{ ports {{ ip publicPort privatePort isIpPublic }} }} }} }}')["pod"]["runtime"]
        for p in (rt or {}).get("ports") or []:
            if p["privatePort"] == 22 and p["isIpPublic"]:
                return p["ip"], p["publicPort"]
        time.sleep(10)
    return None


def ssh_args(st):
    return ["-p", str(st["port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", f"root@{st['ip']}"]


def sh(st, cmd, check=True):
    for _ in range(5):
        rc = subprocess.run(["ssh", *ssh_args(st), "bash", "-lc", shlex.quote(f"mkdir -p /root/lp && cd /root/lp && {cmd}")],
                            stdin=subprocess.DEVNULL).returncode
        if rc != 255:
            break
        time.sleep(15)
    if check and rc:
        raise SystemExit(f"remote command failed rc={rc}: {cmd}")
    return rc


def push(st):
    ssh = "ssh " + " ".join(shlex.quote(a) for a in ssh_args(st))
    subprocess.run(f"tar czf - -C {shlex.quote(ROOT)} --exclude=pods --exclude=__pycache__ echo subliminal_pilot/*.py subliminal_pilot/*.json "
                   f"| {ssh} 'mkdir -p /root/lp && tar xzf - -C /root/lp'", shell=True, check=True)


def load(name):
    return json.load(open(f"{STATE}/{name}.json"))


def main():
    cmd, *rest = sys.argv[1:]
    os.makedirs(STATE, exist_ok=True)
    if cmd == "up":
        name, disk = rest[0], int(rest[1]) if len(rest) > 1 else 150
        for attempt in range(20):
            pod = create(name, disk)
            if not pod:
                print("no capacity; retrying in 60 s", flush=True)
                time.sleep(60)
                continue
            print(f"pod {pod['id']} {pod['gpu']} ({pod['cloud']}) ${pod['costPerHr']}/h", flush=True)
            addr = wait_ssh(pod["id"])
            if addr:
                break
            gql(f'mutation {{ podTerminate(input: {{podId: "{pod["id"]}"}}) }}')
            print("never exposed ssh; terminated, retrying", flush=True)
        else:
            raise SystemExit("could not get a pod")
        st = {"id": pod["id"], "ip": addr[0], "port": addr[1], "cost": pod["costPerHr"], "gpu": pod["gpu"], "t0": time.time()}
        json.dump(st, open(f"{STATE}/{name}.json", "w"))
        time.sleep(5)
        push(st)
        print(f"ready: ssh {' '.join(ssh_args(st))}", flush=True)
    elif cmd == "sh":
        sh(load(rest[0]), rest[1])
    elif cmd == "bg":
        name, log, c = rest
        job = f"({c}) > /root/{log} 2>&1; echo $? > /root/{log}.rc"
        sh(load(name), f"rm -f /root/{log}.rc; nohup setsid bash -c {shlex.quote(job)} >/dev/null 2>&1 </dev/null &")
    elif cmd == "push":
        push(load(rest[0]))
    elif cmd == "pull":
        st = load(rest[0])
        os.makedirs(rest[2], exist_ok=True)
        e = "ssh " + " ".join(shlex.quote(a) for a in ssh_args(st)[:-1])
        subprocess.run(["rsync", "-az", "-e", e, f"{ssh_args(st)[-1]}:{rest[1]}", rest[2]], check=True)
    elif cmd == "down":
        st = load(rest[0])
        gql(f'mutation {{ podTerminate(input: {{podId: "{st["id"]}"}}) }}')
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {rest[0]} after {mins:.0f} min (~${st['cost'] * mins / 60:.2f})")
        os.rename(f"{STATE}/{rest[0]}.json", f"{STATE}/{rest[0]}.done.json")
    elif cmd == "list":
        for f in sorted(os.listdir(STATE)):
            if f.endswith(".json") and not f.endswith(".done.json"):
                st = json.load(open(f"{STATE}/{f}"))
                mins = (time.time() - st["t0"]) / 60
                print(f"{f[:-5]:16s} {st['gpu']:24s} {mins:6.0f} min ~${st['cost'] * mins / 60:.2f}")


if __name__ == "__main__":
    main()
