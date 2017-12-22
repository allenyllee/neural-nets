#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdio.h>
#include "./neural.h"
#include "./data.h"


#define PAR_NUM 3

typedef struct _model1 Model1;
typedef void (*modelInit) (Model1* this, Unit* const p, Data const * const d);
typedef float (*modelForward) (Model1* this);
typedef void (*modelBackward) (Model1* this);
typedef void (*modelUpdateParameter) (Model1* this);


void _init(Model1* this, Unit* const p, Data const * const d);
float _forward(Model1* this);
void _backward(Model1* this);
void _updateParameter(Model1* this);


struct _model1 {
    Unit* parameter;
    Data const * data;
    Unit* out;
    NeuralSet addset, mulset;
    Neural addneu[10], mulneu[10];
    modelInit init;
    modelForward forward;
    modelBackward backward;
    modelUpdateParameter updateParameter;
};

void _init(Model1* this, Unit* const p, Data const * const d){
    init_add(&this->addset, this->addneu);
    init_mul(&this->mulset, this->mulneu);
    this->forward = _forward;
    this->backward = _backward;
    this->updateParameter = _updateParameter;
    this->parameter = p;
    this->data = d;
}

float _forward(Model1* this){
    Unit u[DIM];
    Unit *tmp1 = NULL;

    for (int i=0; i<DIM; i++){
        u[i].value = this->data->x[i];
    }

    // evaluate value
    this->out = this->mulset.forward(&this->mulset.neurals[0], &this->parameter[0], &u[0]); //ax
    tmp1 = this->mulset.forward(&this->mulset.neurals[1], &this->parameter[1], &u[1]); //by
    this->out = this->addset.forward(&this->addset.neurals[0], this->out, tmp1); // ax+by
    this->out = this->addset.forward(&this->addset.neurals[1], this->out, &this->parameter[2]); // ax+by+c

    printf("forward value = %f, label = %d\n", this->out->value, this->data->label);

    return this->out->value;
}

void _backward(Model1* this){
    //int pull=0;
    /*
    this->addset.neurals[1].uout.grad = 0;

    if ( this->data->label == 1 && this->out->value < 1){
        this->addset.neurals[1].uout.grad=1;
    }

    if ( this->data->label == -1 && this->out->value > -1){
        this->addset.neurals[1].uout.grad=-1;
    }
    */
    //printf("pull = %f\n",this->addset.neurals[1].uout.grad);
    // evaluate grad
    this->addset.neurals[1].uout.grad = 1;
    this->addset.backward(&this->addset.neurals[1]);
    this->addset.backward(&this->addset.neurals[0]);
    this->mulset.backward(&this->mulset.neurals[1]);
    this->mulset.backward(&this->mulset.neurals[0]);
}

void _updateParameter(Model1* this){
    float step_size = 0.001;

    if (this->data->label * this->out->value < 0 ){
        step_size *= this->data->label;
    }else {
        step_size = 0;
    }

    //printf("grad = ");
    // update input varible
    for (int i=0; i<PAR_NUM; i++){
        //printf("%f, ", this->parameter[i].grad);
        //this->parameter[i].grad += -this->parameter[i].value; //svm
        this->parameter[i].value += step_size * this->parameter[i].grad;
        this->parameter[i].grad = 0;
    }

    //printf("\n");
}


/**
 * model 1:
 * f(x,y)=ax+by+c
 */
float model1(Unit* const parameter, Data const * const data, int isbackward){
    Unit tmp3;
    Model1 m;

    m.init = _init;
    m.init(&m, parameter, data);

    tmp3.value = m.forward(&m);

    if (isbackward){
        m.backward(&m);
        m.updateParameter(&m);
        // evaluate value
        m.forward(&m);
    }

    return tmp3.value;
}





#endif