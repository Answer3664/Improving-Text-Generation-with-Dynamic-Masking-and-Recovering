import time
from multiprocessing import Process

def f(name):
    print('hello', name)
    time.sleep(1)

if __name__ == '__main__':
    p_lst = []
    for i in range(5):
        p = Process(target=f, args=('bob',))
        p.start()
        p_lst.append(p)
    # [p.join() for p in p_lst]
    print('父进程在执行')