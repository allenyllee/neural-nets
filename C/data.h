#ifndef __DATA_H__
#define __DATA_H__

#define DIM 2

typedef struct _Data Data;
struct _Data {
    float x[DIM];
    int label;
};

#endif