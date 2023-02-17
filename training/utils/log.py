import time

def write_log(file, msg):
    with open(file, 'a+', encoding='utf-8') as f:
        f.write('[' + time.strftime("%Y-%m-%d %H:%M:%S") + ']' + str(msg) + '\n')
