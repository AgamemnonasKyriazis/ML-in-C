#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "data.h"

float **allocate_ndarray(int m, int n)
{
    float **var = (float **)malloc(sizeof(float *) * m);
    for(int i = 0; i < m; i++) 
    {
        var[i] = (float *)malloc(sizeof(float) * n);
        for(int j = 0; j < n; j++)
            var[i][j] = 0;
    }
    return var;
}

int argmax(float **ra) {
    float max_val = ra[0][0];
    int max_indx = 0;
    for(int i = 0; i < 10; i++)
    {
        if(ra[i][0] > max_val)
        {
            max_val = ra[i][0];
            max_indx = i;
        }
    }
    return max_indx;
}

float **add(float **mat1, float **mat2, int m, int n)
{
    float **res = allocate_ndarray(m, n);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            res[i][j] += mat1[i][j] + mat2[i][j];
        }
    }
    return res;
}

float **matmul(float **mat1, int m1, int n1, float **mat2, int m2, int n2)
{
    float **res = allocate_ndarray(m1, n2);
    for(int i = 0; i < m1; i++)
    {
        for(int j = 0; j < n2; j++)
        {
            for(int k = 0; k < n1; k++)
            {
                res[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return res;
}

float **transpose(float **mat, int m, int n)
{
    float **res = allocate_ndarray(n, m);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            res[j][i] = mat[i][j];
    return res;
}

float **apply(float **mat, int m, int n, float (*func)(float))
{
    float **res = allocate_ndarray(m, n);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            res[i][j] = func(mat[i][j]);
    return res;
}

void init_param(float **w1, float **b1, float **w2, float **b2) 
{
    for(int i = 0; i < 10; i++) 
    {
        for(int j = 0; j < 64; j++)
        {
            w1[i][j] = (float)rand()/(float)RAND_MAX;
        }
    }

    for(int i = 0; i < 10; i++) 
    {
        b1[i][0] = (float)rand()/(float)RAND_MAX;
    }

    for(int i = 0; i < 10; i++) 
    {
        for(int j = 0; j < 10; j++)
        {
            w2[i][j] = (float)rand()/(float)RAND_MAX;
        }
    }

    for(int i = 0; i < 10; i++) 
    {
        b2[i][0] = (float)rand()/(float)RAND_MAX;
    }
}

void free_ndarray(float **var, int m)
{
    for(int i = 0; i < m; i++)
    {
        free(var[i]);
    }
    free(var);
    var = NULL;
}

float ReLU(float z)
{
    return (z > 0)? z : 0;
}

float d_ReLU(float z)
{
    return (float)(z >= 0);
}

float logisticf(float z)
{
    return 1 / (1 + expf(-z));
}

float d_logisticf(float z)
{
    return logisticf(z) * (1 - logisticf(z));
}

void forward_prop(float **w1, float **b1, float **w2, float **b2, float **x, float ***z1, float ***a1, float ***z2, float ***a2)
{
    *z1 = matmul(w1, 10, 64, x, 64, 1);
    *z1 = add(*z1, b1, 10, 1);
    *a1 = apply(*z1, 10, 1, ReLU);

    *z2 = matmul(w2, 10, 10, *a1, 10, 1);
    *z2 = add(*z2, b2, 10, 1);
    *a2 = apply(*z2, 10, 1, logisticf);
}

void back_prop(float **z1, float **a1, float **z2, float **a2, float **x, float **y, float ***dw2, float ***db2, float ***dw1, float ***db1, float **w2)
{
    float **dz2 = allocate_ndarray(10, 1);
    for(int i = 0; i < 10; i++)
    {
        dz2[i][0] = -2 * (y[i][0] - a2[i][0]) * d_logisticf(z2[i][0]);
    }
    float **a1T = transpose(a1, 10, 1);
    
    *dw2 = matmul(dz2, 10, 1, a1T, 1, 10);
    for(int i = 0; i < 10; i++)
    {
        (*db2)[i][0] = dz2[i][0];
    }
 
    float **w2T = transpose(w2, 10, 10);
    float **dz1 = matmul(w2T, 10, 10, dz2, 10, 1);
    
    for(int i = 0; i < 10; i++)
        dz1[i][0] = dz1[i][0] * d_ReLU(z1[i][0]);
    

    float **xT = transpose(x, 64, 1);
    *dw1 = matmul(dz1, 10, 1, xT, 1, 64);
    for(int i = 0; i < 10; i++)
        (*db1)[i][0] = dz1[i][0];

    free_ndarray(a1T, 1);
    free_ndarray(w2T, 10);
    free_ndarray(xT, 1);
    free_ndarray(dz2, 10);
    free_ndarray(dz1, 10);
}

void update_param(float **w1, float **b1, float **w2, float **b2, float **dw1, float **db1, float **dw2, float **db2, float h)
{
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 64; j++)
        {
            w1[i][j] = w1[i][j] - h*dw1[i][j];
        }
    }

    for(int i = 0; i < 10; i++)
    {
        b1[i][0] = b1[i][0] - h*db1[i][0];
    }

    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            w2[i][j] = w2[i][j] - h*dw2[i][j];
        }
    }

    for(int i = 0; i < 10; i++)
    {
        b2[i][0] = b2[i][0] - h*db2[i][0];
    }
}

void delay(int number_of_seconds)
{
    // Converting time into milli_seconds
    int milli_seconds = 1000 * number_of_seconds;
 
    // Storing start time
    clock_t start_time = clock();
 
    // looping till required time is not achieved
    while (clock() < start_time + milli_seconds)
        ;
}

int main() 
{
    srand((unsigned int)time(NULL));

    float **w1 = allocate_ndarray(10, 64);
    float **b1 = allocate_ndarray(10, 1);
    float **w2 = allocate_ndarray(10, 10);
    float **b2 = allocate_ndarray(10, 1);

    float **z1 = allocate_ndarray(10, 1);
    float **a1 = allocate_ndarray(10, 1);
    float **z2 = allocate_ndarray(10, 1);
    float **a2 = allocate_ndarray(10, 1);
    float **dw1 = allocate_ndarray(10, 64);
    float **db1 = allocate_ndarray(10, 1);
    float **dw2 = allocate_ndarray(10, 10);
    float **db2 = allocate_ndarray(10, 1);

    int true_positives;

    init_param(w1, b1, w2, b2);

    int batch_size = 1;

    float m = (float)1/batch_size;

    int num_batches = 1000 / batch_size;

    int num_epochs = 100;

    for(int i = 0; i < num_epochs; i++)
    {
        true_positives = 0;
        for(int j = 0; j < num_batches; j++)
        {
            int start_index = j * batch_size;
            int end_index = start_index + batch_size;

            for(int b = start_index; b < end_index; b++)
            {
                float **x_train = allocate_ndarray(64, 1);
                for(int c = 0; c < 64; c++)
                {
                    x_train[c][0] = x[b][c];
                }
                float **y_train = allocate_ndarray(10, 1);
                for(int c = 0; c < 10; c++)
                {
                    y_train[c][0] = y[b][c];
                }
                forward_prop(w1, b1, w2, b2, x_train, &z1, &a1, &z2, &a2);
                back_prop(z1, a1, z2, a2, x_train, y_train, &dw2, &db2, &dw1, &db1, w2);

                if(argmax(y_train) == argmax(a2))
                {
                    true_positives++;
                }
                
                free_ndarray(x_train, 64);
                free_ndarray(y_train, 10);

                update_param(w1, b1, w2, b2, dw1, db1, dw2, db2, 0.1 * m);
            }
        }
        printf("Epoch: %d, Accuracy: %f\n", i, (float)true_positives/1000);
    }
    free_ndarray(w1, 10);
    free_ndarray(b1, 10);
    free_ndarray(w2, 10);
    free_ndarray(b2, 10);

    free_ndarray(dw1, 10);
    free_ndarray(db1, 10);
    free_ndarray(dw2, 10);
    free_ndarray(db2, 10);
    free_ndarray(z1, 10);
    free_ndarray(a1, 10);
    free_ndarray(z2, 10);
    free_ndarray(a2, 10);
    return 0;
}