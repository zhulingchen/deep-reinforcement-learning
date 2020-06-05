import os
import subprocess as sp
import numpy as np
import tensorflow as tf



# unmask specific GPU(s) for training and inference
def mask_busy_gpus(leave_unmasked=1, available_mem=1024, random=True,
                   exclude_process=('python', os.path.splitext(os.path.basename(__file__))[0])):
    try:
        if leave_unmasked < 1:
            # do not use GPU if leave_unmasked < 1
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            # query free memories from all GPUs
            cmd_mem_free = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'
            print('Query free memories from all GPUs:', cmd_mem_free)
            mem_free_list = sp.check_output(cmd_mem_free.split()).decode('ascii').split('\n')[:-1]
            mem_free_list = [int(mem.split()[0]) for mem in mem_free_list]
            print('Free memory list (MB):', mem_free_list)
            available_gpu = [i for i, mem in enumerate(mem_free_list) if mem > available_mem]
            # exclude GPUs when they are running some certain processes
            if exclude_process:
                process_running_gpu = []
                for idx_gpu in range(len(mem_free_list)):
                    # query names of processes running on the current GPU
                    cmd_gpu_proc = 'nvidia-smi --query-compute-apps=process_name --format=csv,noheader,nounits --id=%d' % idx_gpu
                    print('Query names of processes running on the GPU index %d:' % idx_gpu, cmd_gpu_proc)
                    gpu_proc_list = sp.check_output(cmd_gpu_proc.split()).decode('ascii').split('\n')[:-1]
                    print('Names of processes running on the GPU index %d:' % idx_gpu, gpu_proc_list)
                    for e in exclude_process:
                        if any([e in proc for proc in gpu_proc_list]) and (idx_gpu not in process_running_gpu):
                            process_running_gpu.append(idx_gpu)
                available_gpu = [i for i in available_gpu if i not in process_running_gpu]
            if len(available_gpu) < leave_unmasked:
                # do not use GPU when insufficient
                print('Warning: Found only %d usable GPU(s) in the system but requested %d GPU(s)' % (len(available_gpu), leave_unmasked))
                print('Use CPU only')
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            else:
                # shuffle available GPU indices
                if random:
                    available_gpu = np.asarray(available_gpu)
                    np.random.shuffle(available_gpu)
                # update CUDA_VISIBLE_DEVICES environment variable
                unmasked_gpu = available_gpu[:leave_unmasked]
                unmasked_gpu_str = ','.join(map(str, unmasked_gpu))
                os.environ["CUDA_VISIBLE_DEVICES"] = unmasked_gpu_str
                print('Left next %d GPU(s) unmasked: [%s] (from %s available)' % (leave_unmasked, unmasked_gpu_str, str(available_gpu)))
    except FileNotFoundError as e:
        raise Exception('"nvidia-smi" is probably not installed. GPUs are not masked. Error on GPU masking:\n%s' % e.output)
    except sp.CalledProcessError as e:
        raise Exception('Error on GPU masking:\n%s' % e.output)
    return tf.test.is_gpu_available()