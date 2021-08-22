#encoding: utf8
from __future__ import division

import os
import logging
import math

logs = set()

def init_log(name, log_dir, level = logging.INFO):
    if (name, level) in logs: return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    format_str = '%(asctime)s-%(levelname)s-%(filename)s#%(lineno)d:%(message)s'
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'log.log')
    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(level)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    logger.addHandler(f_handler)

# init_log('global')

def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    logger = logging.getLogger('global')
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))


def main():
    for i, lvl in enumerate([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]):
        log_name = str(lvl)
        init_log(log_name, lvl)
        logger = logging.getLogger(log_name)
        print('****cur lvl:{}'.format(lvl))
        logger.debug('debug')
        logger.info('info')
        logger.warning('warning')
        logger.error('error')
        logger.critical('critiacal')
if __name__ == '__main__':
    main()