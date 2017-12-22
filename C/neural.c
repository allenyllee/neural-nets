#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIM 2
#define PAR_NUM 3
#define DATASIZE 6

typedef struct _Unit Unit;
typedef struct _Neural Neural;
typedef struct _NeuralSet NeuralSet;
typedef Unit* (*ForwardFunc) (Neural*, Unit*, Unit*);
typedef void (*BackwardFunc) (Neural*);

typedef struct _Data Data;

struct _Data {
    float x[DIM];
    int label;
};

struct _Unit{
    double value;
    double grad;
};

struct _Neural {
    Unit* u1;
    Unit* u2;
    Unit uout;
};


struct _NeuralSet {
    Neural* neurals;
    ForwardFunc forward;
    BackwardFunc backward;
};


/**
 * add gate
 */
Unit* add(Neural* const this, Unit* a, Unit* b){
    this->u1 = a;
    this->u2 = b;
    this->uout.value = this->u1->value + this->u2->value;
    return &this->uout;
}

void addgrad(Neural* const this){
    this->u1->grad += 1.0 * this->uout.grad;
    this->u2->grad += 1.0 * this->uout.grad;
}

void init_add(NeuralSet* const x, Neural* n){
    x->neurals = n;
    x->forward = add;
    x->backward = addgrad;
}

/**
 * multiply gate
 */
Unit* mul(Neural* const this, Unit* a, Unit* b){
    this->u1 = a;
    this->u2 = b;
    this->uout.value = this->u1->value * this->u2->value;
    return &this->uout;
}

void mulgrad(Neural* const this){
    this->u1->grad += this->u2->value * this->uout.grad;
    this->u2->grad += this->u1->value * this->uout.grad;
}

void init_mul(NeuralSet* const x, Neural* n){
    x->neurals = n;
    x->forward = mul;
    x->backward = mulgrad;
}


/**
 * model 1:
 * f(x,y)=ax+by+c
 */
float model1(Unit* const parameter, Data const * const data, int isbackward){
    // function ax+by+c
    NeuralSet addset, mulset;
    Neural addneu[10], mulneu[10];
    Unit u[DIM];
    Unit *tmp1 = NULL, *tmp2 = NULL;
    Unit tmp3;
    float step_size = 0.001;

    for (int i=0; i<DIM; i++){
        u[i].value = data->x[i];
    }

    init_add(&addset, addneu);
    init_mul(&mulset, mulneu);

    // evaluate value
    tmp1 = mulset.forward(&mulset.neurals[0], &parameter[0], &u[0]); //ax
    tmp2 = mulset.forward(&mulset.neurals[1], &parameter[1], &u[1]); //by
    tmp1 = addset.forward(&addset.neurals[0], tmp1, tmp2); // ax+by
    tmp1 = addset.forward(&addset.neurals[1], tmp1, &parameter[2]); // ax+by+c

    printf("forward value = %f, label = %d\n", tmp1->value, data->label);

    tmp3.value = tmp1->value;

    if (isbackward){

        if (tmp1->value * data->label < 0) {
            // evaluate grad
            addset.neurals[1].uout.grad = 1;
            addset.backward(&addset.neurals[1]);
            addset.backward(&addset.neurals[0]);
            mulset.backward(&mulset.neurals[1]);
            mulset.backward(&mulset.neurals[0]);

            // update input varible
            step_size*=data->label;
            for (int i=0; i<PAR_NUM; i++){
                parameter[i].value += step_size * parameter[i].grad;
                parameter[i].grad = 0;
            }

            // evaluate value
            tmp1 = mulset.forward(&mulset.neurals[0], &parameter[0], &u[0]); //ax
            tmp2 = mulset.forward(&mulset.neurals[1], &parameter[1], &u[1]); //by
            tmp1 = addset.forward(&addset.neurals[0], tmp1, tmp2); // ax+by
            tmp1 = addset.forward(&addset.neurals[1], tmp1, &parameter[2]); // ax+by+c

            printf("forward value = %f\n", tmp1->value);
        }

    }

    return tmp3.value;
}

int main(){
    int count = 0;
    Unit a[PAR_NUM];
    Data d[DATASIZE] = {{.x[0]=1.2, .x[1]=0.7, .label=1},
                {.x[0]=-0.3, .x[1]=0.5, .label=-1},
                {.x[0]=-3, .x[1]=-1, .label=1},
                {.x[0]=0.1, .x[1]=1.0, .label=-1},
                {.x[0]=3.0, .x[1]=1.1, .label=-1},
                {.x[0]=2.1, .x[1]=-3, .label=1}};

    for(int i=0; i<PAR_NUM; i++){
        a[i].value = 1;
        a[i].grad = 0;
    }

    int j=0;
    int i=0;
    srand(time(NULL));

    while (1){
        printf("loop %d\n",j);
        i = rand() % DATASIZE ;
        if ( model1(a,&d[i],1) * d[i].label <0 ){
            count = 0;
        }else {
            count += 1;
            if(count == DATASIZE*5) break;
        }
        j++;
    }

    printf("=============\n");

    for (int i=0; i<DATASIZE; i++){
        model1(a,&d[i],0);
    }

    return 0;
}