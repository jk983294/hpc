#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include "utils/Timer.h"

/**
 * when sleep, the page cool down, so the performance is not hot page performance
 * also notice max is not accurate since we have std::cout operation
 * average is 200ns ~ 600ns
 */

int main(void) {
    int shmid;
    key_t key = 5678;
    uint64_t *shm;

    if (!fork()) {
        // create the segment
        if ((shmid = shmget(key, 100, IPC_CREAT | 0666)) < 0) {
            perror("shmget");
            exit(1);
        }

        // attach the segment to our data space
        if ((shm = (uint64_t *)shmat(shmid, NULL, 0)) == nullptr) {
            perror("shmat");
            exit(1);
        }

        while (1) {
            *shm = ztool::ntime();
            usleep(10000);
        }
    } else {
        sleep(1);

        uint64_t delta;
        uint64_t max = 0;
        uint64_t min = UINT64_MAX;
        uint64_t sum = 0;
        uint64_t count = 0;

        // locate the segment
        if ((shmid = shmget(key, 100, 0666)) < 0) {
            perror("shmget");
            exit(1);
        }

        // attach the segment to our data space
        if ((shm = (uint64_t *)shmat(shmid, NULL, 0)) == nullptr) {
            perror("shmat");
            exit(1);
        }

        uint64_t preTime = *shm;
        while (1) {
            while (preTime == *shm) {
            }
            preTime = *shm;

            delta = ztool::ntime() - preTime;

            if (delta > max)
                max = delta;
            else if (delta < min)
                min = delta;

            sum += delta;
            count++;

            if (count % 1000 == 0) {
                cout << std::fixed << setprecision(3) << "min: " << min << " max: " << max
                     << " avg: " << (sum / (double)count) << " ns" << endl;
                max = 0;
                min = UINT64_MAX;
                sum = 0;
                count = 0;
            }
        }
    }

    return 0;
}
