import os
import hashlib
import dataclasses

RESULTS_DIR = "results"


def get_results_filename(id: str, n_executions: int):
    results_filename = "timing_"
    results_filename += id
    results_filename += f"_exec_{n_executions}"
    results_filename += ".pickle"
    results_filename = os.path.join(RESULTS_DIR, results_filename)
    return results_filename

def get_branch_name():
    return os.popen("git rev-parse --abbrev-ref HEAD").read().strip()

def get_acados_branch_name():
    backup_dir = os.getcwd()
    os.chdir(os.getenv("ACADOS_INSTALL_DIR"))
    branch_name = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()
    os.chdir(backup_dir)
    return branch_name

def dataclass_to_string(dataclass):
    return "_".join([f"{key}_{value}" for key, value in dataclass.__dict__.items()])

def hash_id(label_str: str) -> str:
    hash_str = str(int(hashlib.md5(label_str.encode()).hexdigest(), 16))
    return hash_str


def get_varying_fields(instances: list):
    varying_fields = []
    field_names = [f.name for f in dataclasses.fields(instances[0])]
    for k in field_names:
        for inst in instances:
            if getattr(inst, k) != getattr(instances[0], k):
                varying_fields.append(k)
                break
    constant_fields = list(set(field_names) - set(varying_fields))
    return varying_fields, constant_fields
