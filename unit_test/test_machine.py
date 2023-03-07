'''
Class: Machine
def:
    __init__
    get_color
    allocate_job
    time_proceed
    show
'''

from platform import machine
import Machine


def test_Machine_init():
    """
    Purpose: 
    """
    machine = Machine.Machine(2, 20, 15, 40)
    machine.show()
# end def
