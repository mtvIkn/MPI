#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <fstream>
#include <memory.h>

#define NITER 30000
#define STEPITER 1000

#define GDF(i,j) ((i)+(j)*(NX+2))

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int i, j, n, m;
    int NX = 128;
    int NY = 128;
    double* f, * df;
    FILE* fpw, * fin;

    int ny;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start, stop;

    int jbeg = 0;
    int jend = (NY) / size - 1;
    if (rank == (size - 1) || rank == 0) {
        jend++;
    }

    f = (double*)calloc((NX + 2) * (jend + 1) * sizeof(double));
    df = (double*)calloc((NX + 2) * (jend + 1) * sizeof(double));

    for (j = 0; j <= jend; j++) {
        for (i = 0; i < NX + 2; i++) {
            f[GDF(i, j)] = df[GDF(i, j)] = 0.0;
            if (i == 0) {
                f[GDF(i, j)] = 1.0;
            }
            else if ((rank == 0) && (j == 0)) {
                f[GDF(i, j)] = 1.0;
            }
            else if (i == (NX + 1)) {
                f[GDF(i, j)] = 0.5;
            }
            else if ((rank == size - 1) && (j == jend)) {
                f[GDF(i, j)] = 0.5;
            }
        }
    }

    int mas_size;
    if ((rank == 0) || (rank == size - 1))
        mas_size = NX;
    else
        mas_size = 2 * NX;

    double* send_buff = (double*)calloc(mas_size * sizeof(double));
    double* recv_buff = (double*)calloc(mas_size * sizeof(double));

    int* sendcounts = (int*)calloc(size * sizeof(int));
    int* sdispl = (int*)calloc(size * sizeof(int));
    int* rdispl = (int*)calloc(size * sizeof(int));
    int* recvcounts = (int*)calloc(size * sizeof(int));

    double t0 = MPI_Wtime();

    if (rank != 0) {
        for (int m = 0; m < NX; m++) {
            send_buff[m] = f[GDF(m + 1, jbeg)];
        }
    }
    if (rank == 0) {
        for (int m = 0; m < NX; m++) {
            send_buff[m] = f[GDF(m + 1, jend)];
        }
    }
    else if (rank != size-1) {
        for (int m = NX; m < mas_size; m++) {
            send_buff[m] = f[GDF(m + 1 - NX, jend)];
        }
    }

    int nbeg = 1;
    int nend = NITER;

    int down, up;

    if (rank == (size - 1))
        up = 1;
    else
        up = rank + 1;
    if (rank == 0)
        down = -1;
    else
        down = rank - 1;

    if (down != -1) {
        sendcounts[down] = NX;
        recvcounts[down] = NX;
    }
    if (up != -1) {
        sendcounts[up] = NX;
        recvcounts[up] = NX;
    }
    if ((rank != 0) && (rank != (size - 1))) {
        sdispl[up] += NX;
        rdispl[up] += NX;
    }

    if (size > 1) {
        MPI_Alltoallv(send_buff, sendcounts, sdispl, MPI_DOUBLE, recv_buff, recvcounts, rdispl, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    for (n = 0; n < nend; n++) {
        for (j = 0; j <= jend; j++) {
            memset(sendcounts, 0, size * sizeof(sendcounts));
            memset(recvcounts, 0, size * sizeof(recvcounts));
            memset(rdispl, 0, size * sizeof(rdispl));
            memset(sdispl, 0, size * sizeof(sdispl));

            if (up != -1) {
                sendcounts[up] = NX;
                recvcounts[up] = NX;
            }
            if (down != -1) {
                sendcounts[down] = NX;
                recvcounts[down] = NX;
            }
            if ((rank != 0) && (rank != (size - 1))) {
                sdispl[up] += NX;
                rdispl[up] += NX;
            }

            for (i = 0; i < NX + 1; i++) {
                if ((j == jend) && (up != -1)) {
                    send_buff[sdispl[up] + i - 1] = f[GDF(i, j)];
                    df[GDF(i, j)] = (f[GDF(i + 1, j)] + f[GDF(i - 1, j)] + f[GDF(i, j - 1)] + recv_buff[rdispl[up] + i - 1]) * 0.25 - f[GDF(i, j)];
                }
                else if ((j == jbeg) && (down != -1)) {
                    send_buff[i - 1] = f[GDF(i, j)];
                    df[GDF(i, j)] = (f[GDF(i + 1, j)] + f[GDF(i - 1, j)] + f[GDF(i, j + 1)] + recv_buff[i - 1]) * 0.25 - f[GDF(i, j)];
                }
                else {
                    df[GDF(i, j)] = (f[GDF(i, j + 1)] + f[GDF(i - 1, j)] + f[GDF(i, j - 1)] + f[GDF(i + 1, j)]) * 0.25 - f[GDF(i, j)];
                }
            }

            if ((rank == (size - 1) && (j == jend - 1))||(j==jend)) {
                MPI_Alltoallv(send_buff, sendcounts, sdispl, MPI_DOUBLE, recv_buff, recvcounts, rdispl, MPI_DOUBLE, MPI_COMM_WORLD);
            }                
        }

        for (i = 1; i < (NX + 1); i++) {
            for (j = jbeg; j <= jend; j++) {
                f[GDF(i, j)] += df[GDF(i, j)];
            }
        }
    }

    double t1 = MPI_Wtime();
    double total = t1 - t0;
    printf("Time %d, Process %f\n", rank, total);

    fpw = fopen("progrev.txt", "w");
    for (j = 0; j <= jend; j++) {
         for (i = 0; i < NX + 2; i++) {
             fprintf(fpw, "%.2f", f[GDF(i, j)]);
             if (i == NX + 1) {
                 fprintf(fpw, "\n");
             }
         }
    }

    free(f);
    free(df);
    free(send_buff);
    free(recv_buff);
    free(sendcounts);
    free(recvcounts);
    free(sdispl);
    free(rdispl);
    MPI_Finalize();
    printf("end of program\n");
    return 0;
}



