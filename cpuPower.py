from ctypes import *
from turtle import delay


class INTELCPU:
    def __init__(self):
        self.lib = CDLL('./EnergyLib64.dll')
        r = self.lib.IntelEnergyLibInitialize()
        self.lib.ReadSample()

    # def __del__(self):

    def get_power(self):
        self.lib.ReadSample()
        nmsr = c_int()
        self.lib.GetNumMsrs(byref(nmsr))
        # print(nmsr.value)

        name = create_unicode_buffer(128)

        for i in range(0, nmsr.value):
            funcID = c_int()
            self.lib.GetMsrFunc(i, byref(funcID))
            if funcID.value:
                self.lib.GetMsrFunc(i, byref(funcID))
                self.lib.GetMsrName(i, name)

                print(i, len(name.value), name.value)
                if name.value == 'Processor':
                    double = (c_double * 3)()
                    n = c_int()
                    self.lib.GetPowerData(0, i, byref(double), byref(n))
                    return double[0]
                    # for j in range(0, n.value):
                    #    print(j, double[j])

        return 0


intel = INTELCPU()
intel.get_power()
