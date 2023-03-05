import numpy as np
# TODO - Test


def get_parker_action(machine, job_slot):
    align_score = 0
    act = len(job_slot.slot)

    for i in range(len(job_slot.slot)):
        new_job = job_slot.slot[i]
        if new_job is not None:  # 存在添加的任务

            avail_res = machine.avail_slot[:new_job.len, :]
            res_left = avail_res - new_job.res_vec

            if np.all(res_left[:] >= 0):

                tmp_align_score = avail_res[0, :].dot(new_job.res_vec)

                if tmp_align_score > align_score:
                    align_score = tmp_align_score
                    act = i
    return act


def get_sjf_action(machine, job_slot):
    sjf_score = 0
    act = len(job_slot.slot)  # 如果没有动作,保持

    for i in range(job_slot.slot):
        new_job = job_slot.slot[i]
        if new_job is not None:
            avail_res = machine.avail_slot[:new_job.len, :]
            res_left = avail_res - new_job.res_vec

            if np.all(res_left[:] >= 0):  # 资源足够
                tmp_sjf_score = 1/float(new_job.len)

                if tmp_sjf_score > sjf_score:
                    sjf_score = tmp_sjf_score
                    act = i
    return act


def get_packer_sjf_action(machine, job_slot, knob):
    combined_score = 0
    act = len(job_slot.slot)

    for i in range(len(job_slot.slot)):
        new_job = job_slot.slot[i]
        if new_job is not None:
            avail_res = machine.avail_slot[:new_job.len, :]
            res_left = avail_res - new_job.res_vec

            if np.all(res_left[:] >= 0):
                tmp_align_score = avail_res[0, :].dot(new_job.res_vec)
                tmp_sjf_score = 1/float(new_job.len)
                tmp_combined_score = knob * \
                    tmp_align_score + (1-knob)*tmp_sjf_score

                if tmp_combined_score > combined_score:
                    combined_score = tmp_combined_score
                    act = i
    return act


def get_random_action(machine, job_slot):
    num_act = len(job_slot.slot)+1  # 允许空动作
    act = np.random.randint(num_act)
    return act
