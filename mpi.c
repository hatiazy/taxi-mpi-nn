// mpicc -O3 -lm mpi.c -o nn_mpi
// mpirun -np 8 ./nn_mpi 0 32 1 0.01

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define M_FEATURES 35   // Total number of features
#define N_HIDDEN 64     // Number of hidden layer nodes
#define EPOCHS 20       // Maximum number of training rounds
#define LR 0.01         // Default learning rate
#define BATCH_SIZE 512  // batch size

// Activation function type
#define ACTIVATION_SIGMOID 0
#define ACTIVATION_RELU 1
#define ACTIVATION_TANH 2

// Activation function
double sigmoid(double z) { 
    return 1.0 / (1.0 + exp(-z)); 
}

double relu(double z) {
    return z > 0 ? z : 0;
}

double tanh_activation(double z) {
    return tanh(z);
}

/// Derivative of activation function function
double sigmoid_derivative(double a) {
    return a * (1 - a);
}

double relu_derivative(double a) {
    return a > 0 ? 1.0 : 0.0;
}

double tanh_derivative(double a) {
    return 1 - a * a;
}

// Forward
double forward(double *x, int m, int n, double **W, double *b, double *v, double c, int activation_type) {
    double *hidden = (double *)malloc(n * sizeof(double));
    for (int j = 0; j < n; j++) {
        double z = b[j];
        for (int k = 0; k < m; k++) z += W[j][k] * x[k];
        
        if (activation_type == ACTIVATION_SIGMOID) {
            hidden[j] = sigmoid(z);
        } else if (activation_type == ACTIVATION_RELU) {
            hidden[j] = relu(z);
        } else if (activation_type == ACTIVATION_TANH) {
            hidden[j] = tanh_activation(z);
        }
    }
    double out = c;
    for (int j = 0; j < n; j++) out += v[j] * hidden[j];
    free(hidden);
    return out;
}

// Shuffle local data
void shuffle_local_data(double **X, double *Y, int N, int m, int rank, int epoch) {
    // Default random seed: 123
    // Set a random seed and use rank and epoch to make the shuffle of different processes and epochs different
    srand(123 + rank + epoch * 1000);
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i+1);
        
        double *temp = X[i];
        X[i] = X[j];
        X[j] = temp;

        double tempY = Y[i];
        Y[i] = Y[j];
        Y[j] = tempY;
    }
}

// MPI train function
void train_mpi(double **X, double *Y, int N, int m, int n, int epochs, int batch_size, double lr,
               double *loss_history, double **W, double *b, double *v, double *c, 
               int rank, int size, int activation_type) {
    srand(123 + rank);

    // Parameter initialization (only performed in rank 0)
    if (rank == 0) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < m; k++) {
                W[j][k] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            }
            b[j] = 0.0;
            v[j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
        *c = 0.0;
    }

    // Broadcast the initial parameters to all processes
    for (int j = 0; j < n; j++) {
        MPI_Bcast(W[j], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&b[j], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v[j], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Early stop parameter
    const double early_stop_delta = 1e-6;
    const int early_stop_patience = 3;
    int stop_counter = 0;
    double prev_loss = 1e12;

    // Create array
    double *hidden = (double*)malloc(n * sizeof(double));
    double *z = (double*)malloc(n * sizeof(double));
    double *grad_v = (double*)malloc(n * sizeof(double));
    double *grad_b = (double*)malloc(n * sizeof(double));
    
    // Gradient cumulative array
    double **accum_grad_W = (double**)malloc(n * sizeof(double*));
    for (int j = 0; j < n; j++) {
        accum_grad_W[j] = (double*)malloc(m * sizeof(double));
    }
    double *accum_grad_b = (double*)malloc(n * sizeof(double));
    double *accum_grad_v = (double*)malloc(n * sizeof(double));
    double accum_grad_c;

    // Calculate the batch quantity of each process
    int local_batches = N / batch_size;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle the local data
        shuffle_local_data(X, Y, N, m, rank, epoch);
        
        // Calculate local loss
        double local_loss = 0.0;
        for (int i = 0; i < N; i++) {
            double *x = X[i];
            double y = Y[i];
            
            for (int j = 0; j < n; j++) {
                double s = b[j];
                for (int k = 0; k < m; k++) s += W[j][k] * x[k];
                
                if (activation_type == ACTIVATION_SIGMOID) {
                    hidden[j] = sigmoid(s);
                } else if (activation_type == ACTIVATION_RELU) {
                    hidden[j] = relu(s);
                } else if (activation_type == ACTIVATION_TANH) {
                    hidden[j] = tanh_activation(s);
                }
            }
            double out = *c;
            for (int j = 0; j < n; j++) out += v[j] * hidden[j];
            
            double err = out - y;
            local_loss += 0.5 * err * err;
        }
        
        // Aggregate the loss of all processes
        double global_loss;
        MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double avg_loss = global_loss / (N * size);
        
        if (rank == 0 && loss_history != NULL) {
            loss_history[epoch] = avg_loss;
            printf("Epoch %d, Loss = %.12f\n", epoch+1, avg_loss);
        }

        // Early stop judgment 
        if (rank == 0) {
            if (fabs(prev_loss - avg_loss) < early_stop_delta) {
                stop_counter++;
                if (stop_counter >= early_stop_patience) {
                    printf("Early stopping at epoch %d\n", epoch+1);
                    int stop_signal = 1;
                    MPI_Bcast(&stop_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    break;
                }
            } else {
                stop_counter = 0;
            }
            prev_loss = avg_loss;
            
            int stop_signal = 0;
            MPI_Bcast(&stop_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            int stop_signal;
            MPI_Bcast(&stop_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (stop_signal) break;
        }

        // Train start
        for (int iter = 0; iter < local_batches; iter++) {
            // Initialize the gradient accumulation
            for (int j = 0; j < n; j++) {
                accum_grad_v[j] = 0.0;
                accum_grad_b[j] = 0.0;
                for (int k = 0; k < m; k++) {
                    accum_grad_W[j][k] = 0.0;
                }
            }
            accum_grad_c = 0.0;

            // Local gradient calculation
            for (int bsz = 0; bsz < batch_size; bsz++) {
                int idx = iter * batch_size + bsz;
                double *x = X[idx];
                double y = Y[idx];

                // Forward
                for (int j = 0; j < n; j++) {
                    double s = b[j];
                    for (int k = 0; k < m; k++) s += W[j][k] * x[k];
                    z[j] = s;
                    
                    if (activation_type == ACTIVATION_SIGMOID) {
                        hidden[j] = sigmoid(s);
                    } else if (activation_type == ACTIVATION_RELU) {
                        hidden[j] = relu(s);
                    } else if (activation_type == ACTIVATION_TANH) {
                        hidden[j] = tanh_activation(s);
                    }
                }
                double out = *c;
                for (int j = 0; j < n; j++) out += v[j] * hidden[j];

                double err = out - y;

                // Backpropagation
                for (int j = 0; j < n; j++) {
                    double derivative;
                    if (activation_type == ACTIVATION_SIGMOID) {
                        derivative = sigmoid_derivative(hidden[j]);
                    } else if (activation_type == ACTIVATION_RELU) {
                        derivative = relu_derivative(hidden[j]);
                    } else if (activation_type == ACTIVATION_TANH) {
                        derivative = tanh_derivative(hidden[j]);
                    }
                    
                    grad_v[j] = err * hidden[j];
                    grad_b[j] = err * v[j] * derivative;
                    
                    for (int k = 0; k < m; k++) {
                        accum_grad_W[j][k] += grad_b[j] * x[k];
                    }
                    accum_grad_v[j] += grad_v[j];
                    accum_grad_b[j] += grad_b[j];
                }
                accum_grad_c += err;
            }

            // Aggregate the gradients of all processes
            for (int j = 0; j < n; j++) {
                MPI_Allreduce(MPI_IN_PLACE, accum_grad_W[j], m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &accum_grad_v[j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &accum_grad_b[j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
            MPI_Allreduce(MPI_IN_PLACE, &accum_grad_c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Update para
            double scale = lr / (batch_size * size);
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    W[j][k] -= scale * accum_grad_W[j][k];
                }
                b[j] -= scale * accum_grad_b[j];
                v[j] -= scale * accum_grad_v[j];
            }
            *c -= scale * accum_grad_c;
        }
    }

    // Release memory
    free(hidden); free(z); free(grad_v); free(grad_b);
    for (int j = 0; j < n; j++) free(accum_grad_W[j]);
    free(accum_grad_W); free(accum_grad_b); free(accum_grad_v);
}

// Calculate SSE
double compute_sse(double **X, double *Y, int N, int m, int n,
    double **W, double *b, double *v, double c, int activation_type,
    int rank, const char* dataset_name) {
    double sse = 0.0;

    // Output the first 5 samples
    printf("Rank %d: %s - First 5 samples predictions:\n", rank, dataset_name);
    for (int i = 0; i < 5 && i < N; i++) {
        double y_pred = forward(X[i], m, n, W, b, v, c, activation_type);
        double err = y_pred - Y[i];
        printf("  Rank %d: %s Sample %d: y_true=%.6f, y_pred=%.6f, err=%.6f\n", 
               rank, dataset_name, i, Y[i], y_pred, err);
        sse += err * err;
    }

    // Calculate the SSE of all samples
    for (int i = 5; i < N; i++) {
        double y_pred = forward(X[i], m, n, W, b, v, c, activation_type);
        double err = y_pred - Y[i];
        sse += err * err;
    }

    printf("Rank %d: %s - Computed SSE for %d samples\n", rank, dataset_name, N);

    return sse;
}

// Read the CSV file and return the sample size
int read_csv(const char *filename, double ***X, double **Y, int m) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { 
        printf("Cannot open file: %s\n", filename);
        return -1;
    }

    // Calculate the number of rows
    int lines = 0;
    char buffer[65536];
    while (fgets(buffer, sizeof(buffer), fp)) {
        lines++;
    }
    rewind(fp);
    
    // Skip the title line
    fgets(buffer, sizeof(buffer), fp);
    lines--; 
    
    // Allocate memory
    *X = (double**)malloc(lines * sizeof(double*));
    *Y = (double*)malloc(lines * sizeof(double));
    
    for (int i = 0; i < lines; i++) {
        (*X)[i] = (double*)malloc(m * sizeof(double));
    }
    
    // Read data
    for (int i = 0; i < lines; i++) {
        if (fgets(buffer, sizeof(buffer), fp) == NULL) break;
        
        char *token = strtok(buffer, ",\n");
        int col = 0;
        
        while (token != NULL && col < m + 1) {
            double val = atof(token);
            if (col == m) {
                (*Y)[i] = val; // The last column is the label
            } else {
                (*X)[i][col] = val;
            }
            token = strtok(NULL, ",\n");
            col++;
        }
    }
    
    fclose(fp);
    return lines;
}

// Release data memory
void free_data(double **X, double *Y, int N) {
    for (int i = 0; i < N; i++) {
        free(X[i]);
    }
    free(X);
    free(Y);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default parameters
    int activation_type = ACTIVATION_SIGMOID;
    int batch_size = BATCH_SIZE;
    int hidden_units = N_HIDDEN;
    double learning_rate = LR;
    
    // Parse command-line parameters
    if (argc > 1) activation_type = atoi(argv[1]);
    if (argc > 2) batch_size = atoi(argv[2]);
    if (argc > 3) hidden_units = atoi(argv[3]);
    if (argc > 4) learning_rate = atof(argv[4]);
    
    // Broadcast parameters to all processes
    MPI_Bcast(&activation_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&batch_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hidden_units, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&learning_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Running with activation: %d, batch size: %d, hidden units: %d, learning rate: %.4f\n", 
               activation_type, batch_size, hidden_units, learning_rate);
    }

    printf("Process %d/%d started\n", rank, size);

    // Read train data
    char train_filename[256];
    sprintf(train_filename, "try/train_part_%d.csv", rank);
    
    double **X_train, *Y_train;
    int N_train = read_csv(train_filename, &X_train, &Y_train, M_FEATURES);
    
    if (N_train <= 0) {
        printf("Process %d: Failed to read training data\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    printf("Process %d: Loaded %d training samples\n", rank, N_train);

    // Read test data
    char test_filename[256];
    sprintf(test_filename, "try/test_part_%d.csv", rank);
    
    double **X_test, *Y_test;
    int N_test = read_csv(test_filename, &X_test, &Y_test, M_FEATURES);
    
    if (N_test <= 0) {
        printf("Process %d: Failed to read test data\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    printf("Process %d: Loaded %d test samples\n", rank, N_test);

    // Save the original data
    int original_N_train = N_train;

    // Directly truncate the data to an integer multiple of the batch size
    N_train = (N_train / batch_size) * batch_size;
    
    // Remove the redundant data
    for (int i = N_train; i < original_N_train; i++) {
        free(X_train[i]);
    }

    printf("Process %d: Using %d training samples after truncation\n", rank, N_train);

    // Calculate the batch quantity
    int local_batches = N_train / batch_size;

    // Initialize model parameters
    double **W = (double**)malloc(hidden_units * sizeof(double*));
    for (int j = 0; j < hidden_units; j++) {
        W[j] = (double*)malloc(M_FEATURES * sizeof(double));
    }
    double *b = (double*)malloc(hidden_units * sizeof(double));
    double *v = (double*)malloc(hidden_units * sizeof(double));
    double c;

    double *loss_history = NULL;
    if (rank == 0) {
        loss_history = (double*)malloc(EPOCHS * sizeof(double));
    }

    // Record the start time
    double start_time = MPI_Wtime();
    
    // Train the model
    train_mpi(X_train, Y_train, N_train, M_FEATURES, hidden_units,
              EPOCHS, batch_size, learning_rate, loss_history, W, b, v, &c, rank, size, activation_type);
    
    // Record the end time
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Training time: %.2f seconds\n", end_time - start_time);
    }

    // Calculate the SSE of the local training set and test set
    double local_train_sse = compute_sse(X_train, Y_train, N_train, M_FEATURES, hidden_units, 
        W, b, v, c, activation_type, rank, "TRAIN");

    double local_test_sse = compute_sse(X_test, Y_test, N_test, M_FEATURES, hidden_units,
        W, b, v, c, activation_type, rank, "TEST");

    // Aggregate the SSE of all processes
    double global_train_sse, global_test_sse;
    MPI_Reduce(&local_train_sse, &global_train_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_test_sse, &global_test_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Calculate global RMSE
    if (rank == 0) {
        double train_rmse = sqrt(global_train_sse / (N_train * size));
        double test_rmse = sqrt(global_test_sse / (N_test * size));
        printf("Training RMSE: %.12f\n", train_rmse);
        printf("Test RMSE: %.12f\n", test_rmse);
        
        // Save the loss history
        FILE *fout = fopen("training_history.csv", "w");
        fprintf(fout, "epoch,loss\n");
        for (int e = 0; e < EPOCHS; e++) {
            fprintf(fout, "%d,%.12f\n", e+1, loss_history[e]);
        }
        fclose(fout);
        
        free(loss_history);
    }

    // Release data memory
    free_data(X_train, Y_train, N_train);
    free_data(X_test, Y_test, N_test);
    
    for (int j = 0; j < hidden_units; j++) free(W[j]);
    free(W); free(b); free(v);

    MPI_Finalize();
    return 0;
}