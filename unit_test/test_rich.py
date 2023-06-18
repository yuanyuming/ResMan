import rich_layout


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
    for i in range(1):
        # time.sleep(0.2)
        jobs = next(iter)

        for job in jobs:
            cl = get_cluster()
            clt = rich_layout.ClusterTable(cl)
            ml.update(job.static_info(), clt.update(), "log")
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
    from pettingzoo.test import api_test, performance_benchmark

    # performance_benchmark(env)
    api_test(env, num_cycles=1)
    return env.parameters.cluster


def test_cluster_table():
    import rich_layout

    cl = get_cluster()
    clt = rich_layout.ClusterTable(cl)
    import rich.console

    cs = rich.console.Console()
    cs.print(clt.update())


def test_machine_slots():
    cluster = get_cluster()
    from rich.console import Console
    from rich.panel import Panel

    import rich_layout

    console = Console()
    slots = [
        rich_layout.MachineSlots(machine.id, machine.slots())
        for machine in cluster.machines
    ]
    panel = rich_layout.ClusterGirds(slots, 4)
    console.print(panel.get_gird())


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
    from rich.console import Console
    from rich.layout import Layout

    layout = Layout()
    layout.split_row(Layout(name="Static", ratio=1), Layout(name="Info", ratio=2))
    layout["Static"].update(static_table.get_table())
    layout["Info"].update(info_table.get_table())
    console = Console()
    console.print(layout)
