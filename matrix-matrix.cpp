#include "mpi.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    // Declare parameters
    int myid, master, numprocs, ierr;
    int anstype, row, arows, acols, brows, bcols, crows, ccols;
    double starttime, stoptime;
    int max_arows, max_acols, max_bcols;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Set parameters
    master = 0;

    max_arows = 2000;
    max_acols = 10000;
    max_bcols = 2000;

    if (myid == 0) {
        // master process first generate random data
        // here we assume a is row major and b is col major
        cout << "Generate data randomly ..." << endl;
        double* a = new double[max_arows * max_acols];
        double* b = new double[max_acols * max_bcols];
        double* c = new double[max_arows * max_bcols];

        memset(a, 1234.5678, max_arows * max_acols * sizeof(double));
        memset(b, 5678.1234, max_acols * max_bcols * sizeof(double));
        
        // broadcast of matrix B to each worker
        starttime = MPI_Wtime();

        cout << "Master process boradcast data" << endl;
        MPI_Bcast(b, max_acols * max_bcols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        int numsent = 0;
        for (int i = 1; i < numprocs; i++) {
            auto* start = a + (i - 1) * max_acols;
            MPI_Send(start, max_acols, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            numsent++;
        }
        
        double* c_row_buffer = new double[max_bcols];

        MPI_Status status;
        for (int i = 0; i < max_arows; i++) {
            MPI_Recv(c_row_buffer, max_bcols, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(c + (status.MPI_TAG-1) * max_bcols, c_row_buffer, max_bcols * sizeof(double));

            // we need to continue sending data
            if (numsent < max_arows) {
                auto start = a + numsent * max_acols;
                MPI_Send(start, max_acols, MPI_DOUBLE, status.MPI_SOURCE, numsent+1, MPI_COMM_WORLD);
                numsent++;
            } else {
                // Send exit signal to worker
                double temp = 1.0;
                MPI_Send(&temp, 1, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            }
        }

        stoptime = MPI_Wtime();

        cout << "The master process took " << stoptime - starttime << " seconds to run." << endl;

        delete [] c_row_buffer;
        delete [] a;
        delete [] b;
        delete [] c;

    } else {
        // receive matrix b from master
        double* b_local = new double[max_acols * max_bcols];
        double* a_local_row = new double[max_acols];
        double* c_local_res = new double[max_bcols];

        MPI_Bcast(b_local, max_acols * max_bcols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cout << "Process " << myid << " received matrix b from master" << endl;

        MPI_Status status;
        while (true) {
            // receive one row from matrix A
            MPI_Recv(a_local_row, max_acols, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == 0) {
                cout << "Process " << myid << " finished job " << endl;
                MPI_Finalize();
                exit(0); // exit when receive the tag is 0
            }

            memset(c_local_res, 0.0, max_bcols * sizeof(double));
            // compute the matrix multiplication manually
            for (int i = 0; i < max_bcols; i++) {
                for (int j = 0; j < max_acols; j++) {
                    c_local_res[i] += b_local[i*max_bcols + j] * a_local_row[j];
                }
            }

            // send result back to master
            MPI_Send(c_local_res, max_bcols, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD);
        }
        
        delete [] c_local_res;
        delete [] a_local_row;
        delete [] b_local;
    }

    MPI_Finalize();

    return 0;
}
