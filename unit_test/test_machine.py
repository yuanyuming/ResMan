'''
Class: Machine
def:
    __init__
    get_color
    allocate_job
    time_proceed
    show
'''


def test_Machine_init():
    """
    Purpose: 
    """
    import Machine
    machine = Machine.Machine(1, 2, 20, [25, 40], [4, 6])
    machine.show()
# end def


def test_caaa():
    """
    Purpose: 
    """
    import Machine
    ss = Machine.SlotShow()
    ss.compute_chart()
# end def
