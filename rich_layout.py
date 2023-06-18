from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table


class JobTable:
    def __init__(self, layout_height):  # Add a parameter for the layout size
        self.deque = deque(maxlen=layout_height)
        self.new_table()
        # Create a console object to print the table

    def new_table(self):
        # Create a table object with a title
        self.table = Table(title="Job Table")
        # Add columns with headers and styles
        self.table.add_column(
            "ID", style="cyan", justify="right"
        )  # Right align this column
        self.table.add_column(
            "Res Vec", style="magenta", justify="right"
        )  # Right align this column
        self.table.add_column(
            "Len", style="green", justify="right"
        )  # Right align this column
        self.table.add_column(
            "Priority", style="yellow", justify="right"
        )  # Right align this column
        self.table.add_column(
            "Budget", style="red", justify="right"
        )  # Right align this column
        self.table.add_column(
            "Restrict", style="blue", justify="right"
        )  # Right align this column

    def add_job(self, data):
        # Add a row with the data dict values
        self.new_table()
        self.deque.append(data)
        for data in self.deque:
            self.table.add_row(
                data["id"],
                data["res_vec"],
                data["len"],
                data["priority"],
                data["budget"],
                data["restrict_machines"],
            )
        return self.table


from rich.console import Console
from rich.table import Table


# Define the classes for machine static info, machine info, and machine table
class MachineStaticInfo:
    # Define the constructor with **kwargs as the argument
    def __init__(self, **kwargs):
        # Assign the values of the kwargs dictionary to the attributes of the class
        self.id = kwargs["id"]
        self.res_slot = kwargs["res_slot"]
        self.cost_vector = kwargs["cost_vector"]

    # Define a method to return a list of the attribute values
    def get_values(self):
        return [str(i) for i in [self.id, self.res_slot, self.cost_vector]]


class MachineInfo:
    # Define the constructor with **kwargs as the argument
    def __init__(self, **kwargs):
        # Assign the values of the kwargs dictionary to the attributes of the class
        self.id = kwargs["id"]
        self.reward = kwargs["reward"]
        self.earning = kwargs["earning"]
        # self.slot = kwargs["slot"]
        self.action = kwargs["action"]
        self.bid = kwargs["bid"]
        self.finished_job_num = kwargs["finished_job_num"]
        self.running_job = kwargs["running_job"]

    # Define a method to return a list of the attribute values
    def get_values(self):
        return [
            str(i)
            for i in [
                self.id,
                f"{self.reward:.2f}",
                f"{self.earning:.2f}",
                # self.slot,
                f"{self.action:.2f}",
                f"{self.bid:.2f}",
                self.finished_job_num,
                self.running_job,
            ]
        ]


class MachineSlots:
    def __init__(
        self, id=0, slots={"Resource 0": "██████████", "Resource 1": "██████████"}
    ) -> None:
        self.slots = slots
        self.title = "Machine " + str(id)
        self.table = Table(
            title=self.title, show_header=False, border_style="none", box=None
        )
        self.table.add_column("Resource", style="cyan", justify="right")
        self.table.add_column("Slot", style="magenta", justify="right")
        for key, values in self.slots.items():
            self.table.add_row(key, values)

    def get_table(self):
        return self.table


class MachineTable:
    # Define the constructor with the title and the column names and styles
    def __init__(self, title, columns):
        # Create a table object with the given title
        self.table = Table(title=title)

        # Loop through the columns and add them to the table object
        for column in columns:
            # Get the name and style of each column
            name = column["name"]
            style = column["style"]

            # Add the column to the table object with the given name and style
            self.table.add_column(name, style=style)

    # Define a method to add rows to the table from a list of machine objects
    def add_rows(self, machines):
        # Loop through the machines and get their values as lists
        for machine in machines:
            values = machine.get_values()

            # Add a row to the table object with the values of the machine object
            self.table.add_row(*values)

    # Define a method to print the table to the console
    def print(self):
        # Create a console object
        console = Console()

        # Print the table object to the console
        console.print(self.table)

    def get_table(self):
        return self.table


# Define the subclasses of MachineTable for the static table and the info table
class MachineStaticTable(MachineTable):
    # Define the constructor with no arguments
    def __init__(self):
        # Call the parent constructor with the title and columns for the static table
        super().__init__(
            title="Machine Static Info",
            columns=[
                {"name": "id", "style": "cyan"},
                {"name": "res_slot", "style": "magenta"},
                {"name": "cost", "style": "green"},
            ],
        )


class MachineInfoTable(MachineTable):
    # Define the constructor with no arguments
    def __init__(self):
        # Call the parent constructor with the title and columns for the info table
        super().__init__(
            title="Machine Info",
            columns=[
                {"name": "id", "style": "cyan"},
                {"name": "reward", "style": "bold red"},
                {"name": "earning", "style": "magenta"},
                # {"name": "slot", "style": "green"},
                {"name": "action", "style": "yellow"},
                {"name": "bid", "style": "red"},
                {"name": "finished", "style": "blue"},
                {"name": "running_job", "style": "white"},
            ],
        )


class ClusterGirds:
    def __init__(self, cluster_slot, columns) -> None:
        self.cluster = cluster_slot
        self.gird = Table.grid(expand=True)
        for i in range(columns):
            self.gird.add_column(justify="center")
        for i in range(int(len(cluster_slot) / columns + 1)):
            if i * columns >= len(cluster_slot) - columns:
                self.gird.add_row(
                    *[
                        MachineSlots(
                            id=i * columns + j,
                            slots=cluster_slot[i * columns + j].slots,
                        ).get_table()
                        for j in range(len(cluster_slot) - i * columns)
                    ]
                )
                break
            if i * columns >= len(cluster_slot):
                break
            self.gird.add_row(
                *[
                    MachineSlots(
                        id=i * columns + j, slots=cluster_slot[i * columns + j].slots
                    ).get_table()
                    for j in range(columns)
                ]
            )

    def get_gird(self):
        return self.gird


class ClusterTable:
    def __init__(self, cluster) -> None:
        self.cluster = cluster
        # 创建一个layout对象
        self.layout = Layout()
        # 将layout上下分割，比例为2:1
        self.layout.split_column(
            Layout(name="top", ratio=3), Layout(name="bottom", ratio=1)
        )
        # 将top部分左右分割，比例为1:1
        self.layout["top"].split_row(
            Layout(name="left", ratio=1), Layout(name="right", ratio=3)
        )

    def update(self):
        static_info = [
            MachineStaticInfo(**machine.static_info())
            for machine in self.cluster.machines
        ]
        info = [MachineInfo(**machine.info()) for machine in self.cluster.machines]
        slot = [
            MachineSlots(machine.id, machine.slots())
            for machine in self.cluster.machines
        ]
        static_table = MachineStaticTable()
        static_table.add_rows(static_info)
        info_table = MachineInfoTable()
        info_table.add_rows(info)
        cluster_gird = ClusterGirds(slot, 5)
        # 将static_table放在left部分
        self.layout["left"].update(static_table.get_table())
        # 将info_table放在right部分
        self.layout["right"].update(info_table.get_table())
        # 将cluster_gird放在bottom部分
        self.layout["bottom"].update(cluster_gird.get_gird())
        # 返回layout对象
        return self.layout


class MyLayout:
    def __init__(self):
        # 创建一个控制台对象
        self.console = Console()
        # 创建一个layout对象
        self.layout = Layout()
        # 将命令行分为上下两个部分，比例为2:1
        self.layout.split(Layout(name="top", ratio=2), Layout(name="bottom", ratio=1))
        # 将上面的layout分为两部分，比例为1:2
        self.layout["top"].split_row(
            Layout(name="Job", ratio=1), Layout(name="Cluster", ratio=2)
        )
        width, height = self.console.size
        self.job_table = JobTable(int(height / 3 * 2) - 7)

    def update(self, Job, Cluster, Log):
        # 给每个layout添加一些内容，并用Panel包裹起来，添加黑框，标题和样式
        self.layout["Job"].update(
            Panel(
                self.job_table.add_job(Job),
                border_style="black",
                title="Job",
                title_align="left",
            )
        )
        self.layout["Cluster"].update(
            Panel(Cluster, border_style="black", title="Cluster", title_align="right")
        )
        self.layout["bottom"].update(
            Panel(Log, border_style="black", title="Log", title_align="center")
        )
        return self.layout

    def show(self):
        # 在控制台上显示layout
        self.console.print(self.layout)


if __name__ == "__main__":
    pass
