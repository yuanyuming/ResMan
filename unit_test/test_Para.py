import sys
sys.path.append('c:\\Users\\Yuming\\source\\repos\\ResMan')


def test_import():
    """
    Purpose: test that import works properly, and
    """
    import parameters
    para = parameters.Parameters()
    para.compute_dependent_parameters()
# end def

def test_dist():
    """
    Purpose: test distribution works properly
    """
    import parameters
    para = parameters.Parameters()
    job = para.dist.normal_dist()
    print(job)

    
    
# end def