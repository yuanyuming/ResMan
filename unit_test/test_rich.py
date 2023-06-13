def get_jobs_iter():
    import Environment

    para = Environment.VehicleJobSchedulingParameters()
    return para.job_iterator


def get_layout():
    import rich_layout

    return rich_layout.MyLayout()


def get_job_table():
    import rich_layout

    return rich_layout.JobTable(12)


def test_layout_job_table():
    import time

    iter = get_jobs_iter()
    jt = get_job_table()
    import rich.console

    cs = rich.console.Console()
    for i in range(30):
        time.sleep(0.2)
        job = next(iter)[0]
        cs.print(jt.add_job(job.static_info()))


def test_layout():
    import time

    iter = get_jobs_iter()
    ml = get_layout()
    import rich.console

    cs = rich.console.Console()
    for i in range(3):
        # time.sleep(0.2)
        jobs = next(iter)
        for job in jobs:
            ml.update(job.static_info(), "cluster", "log")
            ml.show()


def get_machine():
    import Environment

    para = Environment.VehicleJobSchedulingParameters()
    return para.cluster.machines[0]


from platform import machine

from rich.console import Console
from rich.panel import Panel
from scipy import cluster


def test_machine_panel():
    machine = get_machine()
    panel_title = f"Machine {machine.id}"
    my_panel = machine_panel(machine.static_info(), machine.info(), panel_title)
    cs = Console()
    cs.print(my_panel)


def get_cluster():
    import Environment

    para = Environment.VehicleJobSchedulingParameters()
    env = Environment.VehicleJobSchedulingEnvACE(para)
    from pettingzoo.test import api_test

    api_test(env)
    return env.parameters.cluster


def test_machine_info_table():
    from rich_layout import (
        MachineInfo,
        MachineInfoTable,
        MachineStaticInfo,
        MachineStaticTable,
    )

    cluster = get_cluster()
    # Create a list of machine static info objects from the cluster machines
    static_machines = [
        MachineStaticInfo(**machine.static_info()) for machine in cluster.machines
    ]

    # Create a list of machine info objects from the cluster machines
    info_machines = [MachineInfo(**machine.info()) for machine in cluster.machines]

    # Create a MachineStaticTable object for the static table
    static_table = MachineStaticTable()

    # Add rows to the static table from the static machines list
    static_table.add_rows(static_machines)

    # Print the static table to the console
    static_table.print()

    # Create a MachineInfoTable object for the info table
    info_table = MachineInfoTable()

    # Add rows to the info table from the info machines list
    info_table.add_rows(info_machines)

    # Print the info table to the console
    info_table.print()
